import openai
import yfinance as yf
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")


# Function to fetch financial data for a given ticker
def fetch_financial_data(ticker_symbol):
    """
    Fetch and prepare financial data for a given ticker symbol.
    Returns a summarized dictionary of financial data.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        if "usd" in ticker_symbol.lower():
            price_data = ticker.history(period="5y").astype(str).to_string()
            return {"Price Data": price_data[:1000]}
        else:
            company = ticker
            income_statement = company.financials.astype(str).to_string()
            balance_sheet = company.balance_sheet.astype(str).to_string()
            cash_flow = company.cashflow.astype(str).to_string()
            company_overview = company.info
            price_data = company.history(period="5y").astype(str).to_string()

            report = {
                "Website": company_overview.get("website", "N/A"),
                "Industry": company_overview.get("industry", "N/A"),
                "Sector": company_overview.get("sector", "N/A"),
                "Business Summary": company_overview.get("longBusinessSummary", "N/A"),
                "Income Statement": income_statement[:1000],  # Truncate for token limits
                "Balance Sheet": balance_sheet[:1000],  # Truncate for token limits
                "Cash Flow": cash_flow[:1000],  # Truncate for token limits
                "Price Data": price_data[:1000],  # Truncate for token limits
            }
            return report
    except Exception as e:
        return {"Error": f"Error fetching data for {ticker_symbol}: {e}"}


# Function to ask OpenAI about financial data
def ask_openai_about_data(financial_data, user_question):
    """
    Ask OpenAI a question based on financial data.
    """
    try:
        # Prepare the prompt
        prompt = (
            f"The following is summarized financial data for a company:\n\n"
            f"Website: {financial_data.get('Website', 'N/A')}\n"
            f"Industry: {financial_data.get('Industry', 'N/A')}\n"
            f"Sector: {financial_data.get('Sector', 'N/A')}\n"
            f"Business Summary: {financial_data.get('Business Summary', 'N/A')}\n"
            f"Income Statement: {financial_data.get('Income Statement', 'Truncated')}\n"
            f"Balance Sheet: {financial_data.get('Balance Sheet', 'Truncated')}\n"
            f"Cash Flow: {financial_data.get('Cash Flow', 'Truncated')}\n"
            f"Price Data: {financial_data.get('Price Data', 'Truncated')}\n\n"
            f"Answer the following question based on the data:\n{user_question}"
        )

        # Call OpenAI's API
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use "gpt-3.5-turbo" if needed
            messages=[
                {"role": "system", "content": "You are a financial analyst providing detailed and accurate answers."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,  # Limit the response size
            temperature=0.7,
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error interacting with OpenAI: {e}"
