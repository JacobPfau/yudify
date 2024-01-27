import json
from openai import OpenAI

client = OpenAI()

def get_response_and_logprob(test_query):
    # Load the JSON file
    with open('/Users/jacobpfau/NYU/other/yud.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Extract queries and responses
    query_list = [item['query'] for item in data]
    response_list = [item['response'] for item in data]

    # Initialize the OpenAI client

    # Constructing the messages for few-shot learning
    messages = [{'role':'system', 'content': 'For each of the following query snippets, rewrite it in the style of Eliezer Yudkowsky.'}]
    for query, response in zip(query_list, response_list):
        messages.append({"role": "user", "content": query})
        messages.append({"role": "assistant", "content": response})

    # Adding the test query
    messages.append({"role": "user", "content": test_query})

    # API call for chat completion
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        logprobs=True,
        top_logprobs=1
    )

    # Extracting the response and logprob
    final_response = completion.choices[0].message.content
    logprob = completion.choices[0].logprobs.content

    return final_response, logprob

test_query = "I think that in the coming 15-30 years, the world could plausibly develop “transformative AI”: AI powerful enough to bring us into a new, qualitatively different future, via an explosion in science and technology R&D. This sort of AI could be sufficient to make this the most important century of all time for humanity. The most straightforward vision for developing transformative AI that I can imagine working with very little innovation in techniques is what I’ll call human feedback on diverse tasks (HFDT): Train a powerful neural network model to simultaneously master a wide variety of challenging tasks (e.g. software development, novel-writing, game play, forecasting, etc) by using reinforcement learning on human feedback and other metrics of performance. HFDT is not the only approach to developing transformative AI, and it may not work at all. But I take it very seriously, and I’m aware of increasingly many executives and ML researchers at AI companies who believe something within this space could work soon. Unfortunately, I think that if AI companies race forward training increasingly powerful models using HFDT, this is likely to eventually lead to a full-blown AI takeover (i.e. a possibly violent uprising or coup by AI systems). I don’t think this is a certainty, but it looks like the best-guess default absent specific efforts to prevent it."  # Replace with your test query
responses = []
for _ in range(25):
    response, logprob = get_response_and_logprob(test_query)
    responses.append({'response': response, 'logprob': sum([entry.logprob for entry in logprob]) / len(logprob)})

for i, resp in enumerate(responses):
    print(f"Run {i+1}: Response - {resp['response']}, Logprob - {resp['logprob']}\n")

output_file = '/Users/jacobpfau/NYU/other/responses.json'
with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(responses, file, indent=4)