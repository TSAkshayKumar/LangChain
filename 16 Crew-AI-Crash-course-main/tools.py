from crewai_tools import YoutubeChannelSearchTool,  SerperDevTool

embedder_config = {
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "sentence-transformers/all-MiniLM-L6-v2"
        }
    }
}

yt_tool = YoutubeChannelSearchTool(
    youtube_channel_handle='@krishnaik06',
    config=embedder_config
)
web_tool = SerperDevTool()

