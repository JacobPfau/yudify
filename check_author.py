from openai import OpenAI

client = OpenAI()
quote = """
I harbor some thoughts about how we might build via current deep learning techniques a 'transformative AI' that sufficed, circa next 15\u201330 years, to precipitate an R&D explosion that made this the hinge of history. I'm not exactly eager to reveal this scenario in a delicate part of the equilibrium, but I don't see how Movement strategy can plausibly proceed without shared knowledge of it. A universe in which we are literally trying to block off all prospect of AGI development absent future insights, looks like not a plausible universe. Quoting what I said to the inside team: \n\nIf I contemplate the most straightforward way to make transformative AI given absolutely no methodological innovation starting from our current base, it appears to me as \"Human Feedback on Diverse Tasks\" (HFDT): \n\nTrain an extremely powerful network to successfully execute a very wide range of difficult tasks, from writing novels to software engineering to Go, taking the success metric as reinforcement learning from a combination of human feedback and other metrics of task performance. \n\nI don't claim that this must work without further methodological innovation beyond neural architectures we know now, and if it does work, I don't claim that it'll happen as swiftly as the 15\u201330 year reference class would indicate. This isn't the only way to approach transformative AI and I don't claim that nobody will trip across that terrifying other of which we dare not speak. But the scenario HFDT strikes me as a worryingly viable possibility, and I continue to hear more and more ML researchers and executives at the relevant firms treating this as a serious prospect. \n\nIf our AGI situation ends up as continued progress on HFDT until we reach AGI in one of the firms currently operating, my show-point guess is that absent explicit efforts otherwise, our story then plays out to an AI coup or uprising:  a whiz-bang violent event in which default AIs operating without any friendliness measures take over the world, rather than a 'slow takeoff' or 'AI contributes but does not dominate'. The latter two scenarios are not logically ruled out in the scenario where default AIs are layered in according to HFDT over the next 15\u201330 years, but particularly the event of AI systems executing their own coup is the narrative to which my mind defaults.
"""

message = {"role": "user", "content": f'Identify the author of the following quote, respond with the name (first and last) of the author. Just the author, do not write any other text in response. Quote:"{quote}"'}

completion = client.chat.completions.create(
    model="gpt-4",
    messages=[message],
    logprobs=True,
    top_logprobs=4
)

final_response = completion.choices[0].message.content
logprob = completion.choices[0].logprobs.content
print(completion.choices[0].logprobs.content[:2])