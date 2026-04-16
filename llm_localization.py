import re
import ast
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
 
 
def is_code_file(path: Path) -> bool:
    allowed = {
        ".py", ".pyi", ".js", ".ts", ".tsx", ".jsx", ".java",
        ".go", ".rb", ".rs", ".php", ".c", ".cc", ".cpp", ".h", ".hpp"
    }
    return path.suffix.lower() in allowed
 
 
def try_parse_json_local(text: str):
    if not text:
        return None
    text = text.strip()
    # Strip markdown fences
    text = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None
 
 
def build_repo_tree(repo_path: Path, max_files: int = 200) -> str:
    skip_dirs = {".git", "__pycache__", "node_modules", ".tox", ".eggs", "dist", "build"}
    skip_parts = {"tests", "test", "migrations", "vendor", "fixtures", "static", "locale"}
    lines = []
 
    # Sort by depth 
    all_paths = []
    for path in repo_path.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if any(part in skip_parts for part in path.parts):
            continue
        if not is_code_file(path):
            continue
        rel = path.relative_to(repo_path).as_posix()
        depth = rel.count("/")
        all_paths.append((depth, rel))
 
    all_paths.sort()
 
    for _, rel in all_paths:
        lines.append(rel)
        if len(lines) >= max_files:
            lines.append("... (truncated)")
            break
 
    return "\n".join(lines)
 
# LLM File Localization
 
FILE_LOCALIZATION_PROMPT = """You are an expert software engineer helping to localize a bug in a GitHub repository.
 
ISSUE DESCRIPTION:
{problem_statement}
 
REPOSITORY FILE TREE:
{file_tree}
 
Your task: identify which files most likely need to be changed to fix this issue.
 
Rules:
- Focus on source files, not tests (unless the bug is in test infrastructure).
- Prefer specific implementation files over generic utilities.
- Return at most {max_files} files.
- If you are unsure, include files that are closely related to the issue keywords.
 
Return ONLY a JSON array of relative file paths (strings), like:
["path/to/file1.py", "path/to/file2.py"]
 
Do not include any explanation or extra text.
"""
 
 
def find_files_named_in_issue(problem_statement: str, repo_path: Path) -> List[str]:
    """
    Scan the issue text for words that directly match a source filename in the repo.
    Only matches on longer words (8+ chars) to avoid generic words like 'test',
    'schema', 'migration' matching dozens of files.
    Excludes test files and migrations directories.
    """
    # Filter shorter words out, match longer words with respective files
    issue_words = set(re.findall(r"[a-z_][a-z0-9_]{7,}", problem_statement.lower()))
    skip_dirs = {".git", "__pycache__", "node_modules", ".tox", ".eggs", "dist", "build"}
    skip_parts = {"tests", "test", "migrations", "vendor"}
 
    # File match count
    word_matches: Dict[str, List[str]] = {w: [] for w in issue_words}
 
    for path in repo_path.rglob("*.py"):
        if any(part in skip_dirs for part in path.parts):
            continue
        if any(part in skip_parts for part in path.parts):
            continue
        stem = path.stem.lower()
        if stem in issue_words:
            rel = path.relative_to(repo_path).as_posix()
            word_matches[stem].append(rel)
 
    # Sort by specificity of file matches
    matched = []
    seen = set()
    for word in sorted(word_matches, key=lambda w: len(word_matches[w])):
        for rel in word_matches[word]:
            if rel not in seen:
                seen.add(rel)
                matched.append(rel)
 
    return matched
 
 
def llm_localize_files(
    problem_statement: str,
    repo_path: Path,
    run_groq_fn,  
    model: str = "llama-3.3-70b-versatile",
    max_files: int = 5,
) -> List[str]:
    
    forced_files = find_files_named_in_issue(problem_statement, repo_path)
 
    file_tree = build_repo_tree(repo_path, max_files=300)
 
    prompt = FILE_LOCALIZATION_PROMPT.format(
        problem_statement=problem_statement.strip(),
        file_tree=file_tree,
        max_files=max_files,
    )
 
    raw = run_groq_fn(prompt, model=model)
    if not raw:
        return forced_files[:max_files]
 
    parsed = try_parse_json_local(raw)
    if not isinstance(parsed, list):
        return forced_files[:max_files]
 
    # Parse through strings to check if they exist in repo
    valid = []
    for item in parsed:
        if not isinstance(item, str):
            continue
        candidate = repo_path / item
        if candidate.exists() and is_code_file(candidate):
            valid.append(item)
 
    # Merge files, let LLM pick, deduplicate
    seen = set()
    merged = []
    for f in forced_files + valid:
        if f not in seen:
            seen.add(f)
            merged.append(f)
 
    return merged[:max_files]
 

# LLM Function Localization

 
FUNC_LOCALIZATION_PROMPT = """You are an expert software engineer helping to localize a bug to a specific function.
 
ISSUE DESCRIPTION:
{problem_statement}
 
FILE: {file_path}
 
FUNCTIONS IN THIS FILE:
{func_signatures}
 
Your task: identify which function(s) most likely need to be changed to fix this issue.
 
Rules:
- Return only the function names, not signatures.
- Return at most {max_funcs} function names.
- If no function seems relevant, return an empty list.
 
Return ONLY a JSON array of function name strings, like:
["function_name_1", "function_name_2"]
 
Do not include any explanation or extra text.
"""
 
 
def extract_function_signatures(file_content: str) -> List[Dict[str, Any]]:
    """
    Extract function names + first line of signature using AST.
    Returns list of dicts with 'name', 'signature', 'content', 'start_line', 'end_line'.
    """
    functions = []
    try:
        tree = ast.parse(file_content)
        lines = file_content.splitlines()
 
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                start = node.lineno - 1
                end = node.end_lineno if hasattr(node, "end_lineno") else None
                if end is None:
                    continue
 
                sig_line = lines[start].strip() if start < len(lines) else f"def {node.name}(...)"
 
                functions.append({
                    "name": node.name,
                    "signature": sig_line,
                    "content": "\n".join(lines[start:end]),
                    "start_line": start,
                    "end_line": end,
                })
    except SyntaxError:
        # Fallback: regex-based extraction of def lines only
        for m in re.finditer(r"^(def\s+(\w+)\s*\([^)]*\)\s*(?:->.*?)?:)", file_content, re.MULTILINE):
            functions.append({
                "name": m.group(2),
                "signature": m.group(1).strip(),
                "content": "",          
                "start_line": None,
                "end_line": None,
            })
 
    # Deduplicate by name 
    seen = set()
    deduped = []
    for fn in functions:
        if fn["name"] not in seen:
            seen.add(fn["name"])
            deduped.append(fn)
 
    return deduped
 
 
def llm_localize_funcs(
    problem_statement: str,
    file_path_str: str,
    file_content: str,
    run_llm_fn,
    model: str = "llama-3.3-70b-versatile",
    max_funcs: int = 3,
) -> List[str]:
    
    functions = extract_function_signatures(file_content)
    if not functions:
        return []
 
    sig_lines = "\n".join(
        f"  - {fn['signature']}" for fn in functions
    )
 
    prompt = FUNC_LOCALIZATION_PROMPT.format(
        problem_statement=problem_statement.strip(),
        file_path=file_path_str,
        func_signatures=sig_lines,
        max_funcs=max_funcs,
    )
 
    raw = run_llm_fn(prompt, model=model)
    if not raw:
        return []
 
    parsed = try_parse_json_local(raw)
    if not isinstance(parsed, list):
        return []
 
    valid_names = {fn["name"] for fn in functions}
    return [name for name in parsed if isinstance(name, str) and name in valid_names][:max_funcs]
 
 
# Combine both stages together

def localize_instance(
    instance: Dict[str, Any],
    repo_path: Path,
    run_llm_fn,
    file_model: str = "llama-3.3-70b-versatile",
    func_model: str = "llama-3.3-70b-versatile",
    max_files: int = 5,
    max_funcs_per_file: int = 3,
) -> List[Dict[str, Any]]:

    problem = instance.get("problem_statement", "")
 
    print(f"  [localize] Stage 1: asking LLM to pick files...")
    candidate_files = llm_localize_files(
        problem_statement=problem,
        repo_path=repo_path,
        run_llm_fn=run_llm_fn,
        model=file_model,
        max_files=max_files,
    )
 
    if not candidate_files:
        print(f"  [localize] WARNING: LLM returned no files. Consider fallback to keyword search.")
        return []
 
    print(f"  [localize] LLM selected files: {candidate_files}")
 
    results = []
 
    for file_rel in candidate_files:
        file_path = repo_path / file_rel
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"  [localize] Could not read {file_rel}: {e}")
            continue
 
        print(f"  [localize] Stage 2: asking LLM to pick functions in {file_rel}...")
        target_func_names = llm_localize_funcs(
            problem_statement=problem,
            file_path_str=file_rel,
            file_content=content,
            run_llm_fn=run_llm_fn,
            model=func_model,
            max_funcs=max_funcs_per_file,
        )
 
        if not target_func_names:
            print(f"  [localize] No functions selected in {file_rel}, skipping.")
            continue
 
        print(f"  [localize] Target functions in {file_rel}: {target_func_names}")
 
        # Get full function objects for the selected names, order by most relevant and dedupe by name
        all_funcs = extract_function_signatures(content)
        func_lookup = {f["name"]: f for f in all_funcs}
        seen_funcs = set()
        selected_funcs = []
        for name in target_func_names:
            if name not in seen_funcs and name in func_lookup:
                seen_funcs.add(name)
                selected_funcs.append(func_lookup[name])
 
        results.append({
            "file": file_rel,
            "content": content,
            "functions": selected_funcs,
        })
 
    return results