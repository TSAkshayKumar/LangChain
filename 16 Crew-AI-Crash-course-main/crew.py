from crewai import Crew,Process
from agents import blog_researcher,blog_writer, domain_expert
from tasks import research_task,write_task,validate_task


# Forming the tech-focused crew with some enhanced configurations
crew = Crew(
  agents=[blog_researcher, domain_expert, blog_writer],
  tasks=[research_task, validate_task, write_task],
  process=Process.sequential,  # Optional: Sequential task execution is default
  memory=True,
  cache=True,
  max_rpm=100,
  share_crew=True
)

## start the task execution process with enhanced feedback
result=crew.kickoff(inputs={'topic':'AI VS ML VS DL vs Data Science'})
print(result)