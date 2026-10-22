import urllib.request, json
url = "https://huggingface.co/api/datasets/ynyg/ExamInk-Seg"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        print("Repo found:", data.get("id"))
        siblings = data.get("siblings", [])
        print(f"Total files: {len(siblings)}")
        for f in siblings[:15]:
            print("  ", f["rfilename"])
except Exception as e:
    print("Error:", e)
