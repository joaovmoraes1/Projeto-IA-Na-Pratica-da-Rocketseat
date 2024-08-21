import os
from datetime import datetime

import yfinance as yf

from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

import streamlit as st

# Define the OpenAI API key (não use secrets.toml)
openai_api_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

if openai_api_key is None:
    st.error("OPENAI_API_KEY environment variable is not set.")
else:
    # CRIANDO YAHOO FINANCE TOOL 
    def fetch_stock_price(ticket):
        stock = yf.download(ticket, start="2023-08-08", end="2024-08-08")
        return stock

    yahoo_finance_tool = Tool(
        name="Yahoo Finance Tool",
        description="Fetches stock prices for {ticket} from the last year about a specific company from Yahoo Finance API",
        func=lambda ticket: fetch_stock_price(ticket)
    )

    # IMPORTANDO OPENAI LLM - GPT
    try:
        llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")
    except Exception as e:
        st.error(f"Error creating ChatOpenAI instance: {e}")
    
    stockPriceAnalyst = Agent(
        role="Senior Stock Price Analyst",
        goal="Find the {ticket} stock price and analyze trends",
        backstory="""You're highly experienced in analyzing the price of a specific stock
        and making predictions about its future price.""",
        verbose=True,
        llm=llm,
        max_iter=5,
        memory=True,
        tools=[yahoo_finance_tool],
        allow_delegation=False
    )

    getStockPrice = Task(
        description="Analyze the stock {ticket} price history and create a trend analysis of up, down, or sideways",
        expected_output="""Specify the current trend stock price - up, down, or sideways. 
        eg. stock= 'AAPL, price UP'""",
        agent=stockPriceAnalyst
    )

    # IMPORTANT: TOOL DE SEARCH 
    search_tool = DuckDuckGoSearchResults(backend='news', num_results=10)

    newsAnalyst = Agent(
        role="Stock News Analyst",
        goal="""Create a short summary of the market news related to the stock {ticket} company. Specify the current trend - up, down, or sideways with
        the news context. For each request stock asset, specify a number between 0 and 100, where 0 is extreme fear and 100 is extreme greed.""",
        backstory="""You're highly experienced in analyzing market trends and news and have tracked assets for more than 10 years.

        You're also a master-level analyst in traditional markets and have a deep understanding of human psychology.

        You understand news, their titles, and information, but you look at those with a healthy dose of skepticism. 
        You also consider the source of the news articles.""",
        verbose=True,
        llm=llm,
        max_iter=10,
        memory=True,
        tools=[search_tool],
        allow_delegation=False
    )

    get_news = Task(
        description=f"""Take the stock and always include BTC to it (if not requested).
        Use the search tool to search each one individually. 

        The current date is {datetime.now()}.

        Compose the results into a helpful report""",
        expected_output="""A summary of the overall market and one sentence summary for each request asset. 
        Include a fear/greed score for each asset based on the news. Use format:
        <STOCK ASSET>
        <SUMMARY BASED ON NEWS>
        <TREND PREDICTION>
        <FEAR/GREED SCORE>""",
        agent=newsAnalyst
    )

    stockAnalystWrite = Agent(
        role="Senior Stock Analyst Writer",
        goal="""Analyze the trends, price, and news and write an insightful, compelling, and informative 3-paragraph long newsletter based on the stock report and price trend.""",
        backstory="""You're widely accepted as the best stock analyst in the market. You understand complex concepts and create compelling stories
        and narratives that resonate with wider audiences. 

        You understand macro factors and combine multiple theories - e.g., cycle theory and fundamental analyses. 
        You're able to hold multiple opinions when analyzing anything.""",
        verbose=True,
        llm=llm,
        max_iter=5,
        memory=True,
        allow_delegation=True
    )

    writeAnalyses = Task(
        description="""Use the stock price trend and the stock news report to create an analysis and write the newsletter about the {ticket} company
        that is brief and highlights the most important points.
        Focus on the stock price trend, news, and fear/greed score. What are the near future considerations?
        Include the previous analyses of stock trend and news summary.""",
        expected_output="""An eloquent 3-paragraph newsletter formatted as markdown in an easy-to-read manner. It should contain:

        - 3 bullet points executive summary 
        - Introduction - set the overall picture and spike up the interest
        - Main part provides the meat of the analysis including the news summary and fear/greed scores
        - Summary - key facts and concrete future trend prediction - up, down, or sideways.""",
        agent=stockAnalystWrite,
        context=[getStockPrice, get_news]
    )

    crew = Crew(
        agents=[stockPriceAnalyst, newsAnalyst, stockAnalystWrite],
        tasks=[getStockPrice, get_news, writeAnalyses],
        verbose=2,
        process=Process.hierarchical,
        full_output=True,
        share_crew=False,
        manager_llm=llm,
        max_iter=15
    )

    # Streamlit UI
    with st.sidebar:
        st.header('Enter the Stock to Research')

        with st.form(key='research_form'):
            topic = st.text_input("Select the ticket")
            submit_button = st.form_submit_button(label="Run Research")
            
    if submit_button:
        if not topic:
            st.error("Please fill the ticket field")
        else:
            try:
                results = crew.kickoff(inputs={'ticket': topic})
                st.subheader("Results of research:")
                st.write(results['final_output'])
            except Exception as e:
                st.error(f"Error during crew kickoff: {e}")