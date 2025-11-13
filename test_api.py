import requests
response = requests.post(
  "http://localhost:8088/judge",
  json={
    "type":"python",
    "solution":"print(9)",
    "expected_output":"9"
  })
print(response.json())