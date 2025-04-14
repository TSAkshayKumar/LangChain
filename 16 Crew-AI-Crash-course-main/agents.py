from crewai import Agent
from langchain_community.llms import Ollama
from tools import yt_tool, web_tool
from dotenv import load_dotenv
import os


# Connect to the local Ollama server
load_dotenv()
os.environ['OPENAI_API_KEY'] = "DUMMY KEY" #Mandatory to add this in os with any dummy key
os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY') #For web searching using Serper tool. 

ollama_llm = Ollama(model="ollama/gemma:2b")  # Include the provider prefix in the model name

## Create a senior blog content researcher
blog_researcher=Agent(
    role='Blog Researcher from Youtube Videos',
    goal='get the relevant video transcription for the topic {topic} from the provided Yt channel',
    verbose=True,
    memory=True,
    backstory=(
       "Expert in understanding videos in AI Data Science , Machine Learning, deep learning And GEN AI and providing suggestion" 
    ),
    tools=[yt_tool],
    allow_delegation=True,
    llm=ollama_llm,
)

## Create a senior data scientist
domain_expert=Agent(
    role='Data scientist',
    goal='You will get video transcription for the topic {topic} your task is to validate the content and add extra on those topic from web browswer',
    verbose=True,
    memory=True,
    backstory=(
       "Expert in Data science, Machine Learning, deep learning And GEN AI and providing valueable informatin" 
    ),
    tools=[web_tool],
    allow_delegation=True,
    llm=ollama_llm,
)

## creating a senior blog writer agent with YT tool
blog_writer=Agent(
    role='Blog Writer',
    goal='Narrate compelling tech stories about the video {topic} from YT video',
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft"
        "engaging narratives that captivate and educate, bringing new"
        "discoveries to light in an accessible manner."
    ),
    tools=[],
    llm=ollama_llm,
    allow_delegation=False
)