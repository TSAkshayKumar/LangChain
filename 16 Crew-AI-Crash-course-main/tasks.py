from crewai import Task
from tools import yt_tool, web_tool
from agents import blog_researcher, blog_writer, domain_expert

## Research Task
research_task = Task(
  description=(
    "Identify the video {topic}."
    "Get detailed information about the video from the channel video."
  ),
  expected_output='A comprehensive 3 paragraphs long report based on the {topic} of video content.',
  tools=[yt_tool],
  async_execution=False,
  agent=blog_researcher,
)

#domain expert
validate_task = Task(
    description=(
        "Based of the {topic}."
        "Validate the the provided content and add extra information from internet"
    ),
    expected_output='A comprehensive 3 paragraphs long report based on the {topic}',
    tools=[web_tool],
    async_execution=False,
    agent=domain_expert,
)

# Writing task with language model configuration
write_task = Task(
  description=(
    "get the info from the youtube channel on the topic {topic}."
  ),
  expected_output='Summarize the info from the youtube channel video on the topic{topic} and create the content for the blog',
  tools=[],
  agent=blog_writer,
  async_execution=False,
  output_file='new-blog-post.md'  # Example of output customization
)
