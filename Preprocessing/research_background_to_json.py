import os, json


# YOUR RESEARCH QUESTION HERE
research_question = '''
YOUR RESEARCH QUESTION HERE
'''

# YOUR BACKGROUND SURVEY HERE
background_survey = '''
YOUR BACKGROUND SURVEY HERE
'''


# Save the research question and background survey to a JSON file
with open("./custom_research_background.json", "w") as f:
    json.dump([research_question, background_survey], f, indent=4)