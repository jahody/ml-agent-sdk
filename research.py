from anthropic import Anthropic
import requests
import json
import time

# Semantic Scholar API helper functions with rate limiting
def semantic_scholar_search(query, limit=5):
    """Search Semantic Scholar for papers with rate limiting"""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "paperId,title,abstract,year,citationCount,authors"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            time.sleep(1)  # Rate limiting: wait 1 second between calls
            return response.json()
        elif response.status_code == 429:
            print(f"Rate limited. Waiting 5 seconds...")
            time.sleep(5)
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                time.sleep(1)
                return response.json()
        print(f"API error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Search failed: {e}")
    return {"data": []}

def get_citations(paper_id, limit=3):
    """Get papers that cite this paper"""
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
    params = {
        "limit": limit,
        "fields": "paperId,title,abstract,year,citationCount"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    return {"data": []}

def get_references(paper_id, limit=3):
    """Get papers referenced by this paper"""
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references"
    params = {
        "limit": limit,
        "fields": "paperId,title,abstract,year,citationCount"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    return {"data": []}

# Define the tool for Claude
tools = [
    {
        "name": "find_papers_for_task",
        "description": "Search for academic papers that could be applicable to a specific research task or problem. Returns ranked papers with relevance scores.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_description": {
                    "type": "string",
                    "description": "Detailed description of the research task or problem to solve"
                },
                "max_papers": {
                    "type": "integer",
                    "description": "Maximum number of papers to return (default: 10)",
                    "default": 10
                }
            },
            "required": ["task_description"]
        }
    }
]

# Implementation
def find_papers_for_task(task_description, max_papers=10):
    """
    Find and rank academic papers for a given task.
    Simplified version with fewer API calls to avoid rate limiting.
    """
    print(f"Searching for papers on: {task_description}")

    # 1. Search Semantic Scholar - request more papers upfront
    search_limit = min(max_papers, 10)  # Cap at 10 to avoid rate limits
    papers_response = semantic_scholar_search(task_description, limit=search_limit)

    papers_list = papers_response.get('data', [])

    if not papers_list:
        print("No papers found.")
        return []

    print(f"Found {len(papers_list)} papers. Analyzing relevance...")

    # 2. Use Claude to analyze each paper (no citation expansion to reduce API calls)
    client = Anthropic()

    for paper in papers_list:
        prompt = f"""Given this task: {task_description}

And this paper:
Title: {paper.get('title', 'N/A')}
Abstract: {paper.get('abstract', 'No abstract available')}
Year: {paper.get('year', 'N/A')}
Citations: {paper.get('citationCount', 0)}

Rate 0-10 how well this paper's methods could apply to the task.
Consider:
- Method applicability to the specific task
- Domain similarity
- Technical approach fit
- Results relevance
- Recency and citation impact

Return ONLY valid JSON: {{"score": X, "reasoning": "brief explanation"}}"""

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            analysis = json.loads(response.content[0].text)
            paper['relevance_score'] = analysis['score']
            paper['reasoning'] = analysis['reasoning']
        except Exception as e:
            print(f"Failed to parse analysis for paper: {e}")
            paper['relevance_score'] = 0
            paper['reasoning'] = "Failed to parse"

    # 3. Sort and return top results
    ranked = sorted(papers_list, key=lambda x: x.get('relevance_score', 0), reverse=True)
    top_papers = ranked[:max_papers]

    print(f"Returning top {len(top_papers)} papers.")
    return top_papers

# Agent loop
def run_research_agent():
    client = Anthropic()
    messages = []
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        messages.append({"role": "user", "content": user_input})
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=tools,
            messages=messages
        )
        
        # Handle tool use
        if response.stop_reason == "tool_use":
            tool_use = next(block for block in response.content if block.type == "tool_use")
            
            if tool_use.name == "find_papers_for_task":
                # Execute the tool
                results = find_papers_for_task(**tool_use.input)
                
                # Format results for Claude
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": json.dumps([{
                        "title": p.get('title'),
                        "year": p.get('year'),
                        "citations": p.get('citationCount'),
                        "relevance_score": p.get('relevance_score'),
                        "reasoning": p.get('reasoning'),
                        "paper_id": p.get('paperId')
                    } for p in results], indent=2)
                }
                
                # Continue conversation with tool result
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": [tool_result]})
                
                # Get final response
                final_response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4096,
                    tools=tools,
                    messages=messages
                )
                
                print(f"\nClaude: {final_response.content[0].text}\n")
                messages.append({"role": "assistant", "content": final_response.content})
        else:
            print(f"\nClaude: {response.content[0].text}\n")
            messages.append({"role": "assistant", "content": response.content})

if __name__ == "__main__":
    run_research_agent()