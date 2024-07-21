import streamlit as st
import webbrowser
from breeze_connect import BreezeConnect
import datetime
import datetime as dt
import pandas as pd
from datetime import date, time, datetime, timedelta
import numpy as np
import time
from PIL import Image
from streamlit_option_menu import option_menu
from statsmodels.tsa.ar_model import AutoReg
import plotly.graph_objects as go
import quantstats as qs
import requests
from io import BytesIO
import zipfile
import os
from io import StringIO
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


# Layout and Logo
# Site Icon
logo_top = Image.open("./tradatanalytix logo.png")
st.set_page_config(page_title = 'TraDatAnalytix',layout='wide', page_icon=logo_top)



# Create ICICI Direct Session for getting Market Data
api_key="16G_Mh68829o5105pg1646!O09d2fm43"

session_key = 44581296

breeze = BreezeConnect(api_key="16G_Mh68829o5105pg1646!O09d2fm43")
breeze.generate_session(api_secret="6759%V7C09Acs(3567164*J00x@06`)3",
                            session_token=session_key)


# Getting SYMBOL and STOCK codes needed for display and data fetch
def extract_text_from_zip(url, file_name):
    # Fetch the zip file content from the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Read the content of the zip file
        zip_data = BytesIO(response.content)

        # Create a ZipFile object from the bytes
        zip_file = zipfile.ZipFile(zip_data)

        # Extract the text file you want
        with zip_file.open(file_name) as txt_file:
            # Read the content of the text file
            text_content = txt_file.read()

            # Decode the content (assuming it's UTF-8 encoded)
            decoded_content = text_content.decode('utf-8')

            return decoded_content
    else:
        print("Failed to fetch the zip file.")

# Example usage
url = "https://directlink.icicidirect.com/NewSecurityMaster/SecurityMaster.zip"  # Replace with your actual URL
file_name = "NSEScripMaster.txt"  # Replace with the name of the text file you want to extract

text_content = extract_text_from_zip(url, file_name)


def string_to_dataframe(data_string):
    # Convert the string to a file-like object
    data_file = StringIO(data_string)

    # Read the data into a pandas DataFrame
    df = pd.read_csv(data_file, delimiter=',')

    return df
df = string_to_dataframe(text_content)
#print(df)

master = df[[' "ShortName"',' "Symbol"',' "Series"', ' "CompanyName"', ' "ExchangeCode"']]

eq_base = master[master[' "Series"'] == "EQ"]

#nifty500 = pd.read_csv("NIFTYNEXT50.csv")
nifty500 = pd.read_csv("NIFTY500_1.csv")

df_nf500 = pd.merge(nifty500, eq_base, left_on='SYMBOL', right_on=' "ExchangeCode"', how='inner')

df_nf500_list = df_nf500[[' "ShortName"']]
df_symbol_list = df_nf500[['SYMBOL']].iloc[1:, 0].tolist()

symbolList = df_nf500_list.iloc[1:, 0].tolist()


# Fetch Data for stocks Function

def get_stock_data(sym12):
        nifty = []
        ns = [sym12]
        now = datetime.now() - timedelta(days = 1)
        from_date = datetime.now() - timedelta(days = 720)
        df = breeze.get_historical_data(interval="1day",
                                    from_date= from_date.strftime('%Y-%m-%dT09:20:00.000Z'),
                                    to_date= now.strftime('%Y-%m-%dT15:25:00.000Z'),
                                    stock_code=sym12,
                                    exchange_code="NSE",
                                    product_type="cash")
        data = pd.DataFrame(df['Success'])

        # Convert with a specific format
        data['date_column'] = pd.to_datetime(data['datetime']).dt.strftime('%Y-%m-%d')
        data.set_index('date_column', inplace=True)
        data2 = data["close"].astype(float)
        if not data2.empty:
            nifty.append(data2)

        nifty_prices = pd.concat(nifty, axis = 1)
        nifty_prices.columns = ns
        stock_data_close = nifty_prices[[sym12]]
        return(nifty_prices)






def portfolio_analytics():

    #lc, rc = st.columns(2)


    with st.sidebar:
      selected_option = option_menu(
        "Select:",
        ['Statistics', 'M/L Optimiser', 'Techno-Funda Insights', 'Strategy'],
        icons = ['bar-chart-fill', 'gear', 'currency-exchange'],
        menu_icon = "cast",
        default_index = 0
      )


    if selected_option == 'Statistics':

        genre = st.radio(
            "Upload your Stock Portfolio:",
            ["Select Manually", "Upload CSV"], horizontal = True)

        if genre == "Select Manually":
            stock_select = st.container(height = 130).multiselect("Select Stocks", df_symbol_list , ['IDFC', 'SBIN'])


            lc, rc = st.columns(2)
            df_sel = pd.DataFrame(stock_select, columns=['SYMBOL'])
            df_sel2 = pd.merge(df_sel, eq_base, left_on='SYMBOL', right_on=' "ExchangeCode"', how='inner')
            symbolList = df_sel2[[' "ShortName"']].iloc[0:, 0].tolist()
            #df = pd.DataFrame(columns=['date_column', 'Close'])
            df = get_stock_data("NIFTY")
            for symbol in symbolList:
                df2 = get_stock_data(symbol)
                df3 = pd.merge(df, df2, on = 'date_column', how = 'right')
                df = df3
            df_final = df
            correlation_matrix = df_final.corr(method='pearson')
            #st.write(correlation_matrix)
            #st.write(df_final)
            fig1 = plt.figure()
            sns.heatmap(correlation_matrix, xticklabels=correlation_matrix.columns, yticklabels=correlation_matrix.columns,
            cmap='YlGnBu', annot=True, linewidth=0.5)
            print('Correlation between Stocks in your portfolio')
            lc.pyplot(fig1)

            daily_simple_return = df_final.pct_change(1)
            daily_simple_return.dropna(inplace=True)

            fig2, ax2 = plt.subplots(figsize = (10,5))
            daily_simple_return.plot(kind = "box",ax = ax2, title = "Risk Box Plot")
            rc.pyplot(fig2)



            daily_cummulative_simple_return =(daily_simple_return+1).cumprod()

            #visualize the daily cummulative simple return
            print('Cummulative Returns')
            fig, ax = plt.subplots(figsize=(18,8))

            for i in daily_cummulative_simple_return.columns.values :
                ax.plot(daily_cummulative_simple_return[i], lw =2 ,label = i)

            ax.legend( loc = 'upper left' , fontsize =10)
            ax.set_title('Daily Cummulative Simple returns/growth of investment')
            ax.set_xlabel('Date')
            ax.set_ylabel('Growth of ₨ 1 investment')
            st.pyplot(fig)
            def format_text(x):
                return f'{x:.2f}'

            fig = px.imshow(correlation_matrix, text_auto=lambda x: format_text(x), aspect="auto", color_continuous_scale='Viridis')
            st.plotly_chart(fig)

        else:
            uploaded_file = st.container(height = 130).file_uploader("(OR) Upload your portfolio holdings CSV file", type=["csv"])
        # #stock_select = lc.container(height = 130).multiselect("Select Stocks", df_symbol_list , ['IDFC', 'SBIN'])
        # #uploaded_file = rc.container(height = 130).file_uploader("(OR) Upload your portfolio holdings CSV file", type=["csv"])
        # df_sel = pd.DataFrame(stock_select, columns=['SYMBOL'])
        # df_sel2 = pd.merge(df_sel, eq_base, left_on='SYMBOL', right_on=' "ExchangeCode"', how='inner')
        # symbolList = df_sel2[[' "ShortName"']].iloc[0:, 0].tolist()
        # #df = pd.DataFrame(columns=['date_column', 'Close'])
        # df = get_stock_data("NIFTY")
        # for symbol in symbolList:
        #     df2 = get_stock_data(symbol)
        #     df3 = pd.merge(df, df2, on = 'date_column', how = 'right')
        #     df = df3
        # df_final = df
        # correlation_matrix = df_final.corr(method='pearson')
        # #st.write(correlation_matrix)
        # #st.write(df_final)
        # fig1 = plt.figure()
        # sns.heatmap(correlation_matrix, xticklabels=correlation_matrix.columns, yticklabels=correlation_matrix.columns,
        # cmap='YlGnBu', annot=True, linewidth=0.5)
        # print('Correlation between Stocks in your portfolio')
        # lc.pyplot(fig1)
        #
        # daily_simple_return = df_final.pct_change(1)
        # daily_simple_return.dropna(inplace=True)
        #
        # fig2, ax2 = plt.subplots(figsize = (10,5))
        # daily_simple_return.plot(kind = "box",ax = ax2, title = "Risk Box Plot")
        # rc.pyplot(fig2)
        #

        # extend pandas functionality with metrics, etc.
        #qs.extend_pandas()

        # fetch the daily returns for a stock
        #stock = qs.utils.download_returns('META')

        # show sharpe ratio
        #st.write(qs.stats.sharpe(stock))

    if selected_option == 'Indices Data':

        # Check if a file has been uploaded
        if uploaded_file is not None:
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(uploaded_file)

            # Display the DataFrame
            st.write(df)







def market_insights():
    st.title("Second page")

    with st.sidebar:
      selected_option = option_menu(
        "Market Insights",
        ['Authentication','Global Markets','Indices Data','Open Interest Data', 'FII/DII Data','Pick Outperformers' ,'Trading Strategy'],
        icons = ['globe', 'globe','body-text','bar-chart-fill', 'gear', 'currency-exchange' ,'option'],
        menu_icon = "cast",
        default_index = 0
      )


    if selected_option == 'Indices Data':


        #st.title("Post Budget - Trading Strategy")
        # horizontal menu
        market_index_select = option_menu("Select Index", ["Nifty", "Bank Nifty"],
        icons=['collection', 'bank2'],
        menu_icon="graph-up-arrow", default_index=0, orientation = "horizontal")



        #selected2

        # Test Nifty Dataframe Prices

        sym11 = "DIVLAB"
        sym12 = "NIFTY"


        nifty = []
        ns = ['Close']

        now = datetime.now() - timedelta(days = 1)
        from_date = datetime.now() - timedelta(days = 720)

        df = breeze.get_historical_data(interval="1day",
                                    from_date= from_date.strftime('%Y-%m-%dT09:20:00.000Z'),
                                    to_date= now.strftime('%Y-%m-%dT15:25:00.000Z'),
                                    stock_code=sym12,
                                    exchange_code="NSE",
                                    product_type="cash")

        data = pd.DataFrame(df['Success'])

        # Convert with a specific format
        data['date_column'] = pd.to_datetime(data['datetime']).dt.strftime('%Y-%m-%d')
        data.set_index('date_column', inplace=True)
        data2 = data["close"].astype(float)
        data2.info()
        print(data2)


        if not data2.empty:
            nifty.append(data2)


        nifty_prices = pd.concat(nifty, axis = 1)

        nifty_prices.columns = ns

        #st.write(nifty_prices)

        stock_data_close = nifty_prices[["Close"]]

        # Change frequency to day
        stock_data_close = stock_data_close.asfreq("D", method="ffill")

        # Fill missing values
        stock_data_close = stock_data_close.ffill()

        # Define training and testing area
        train_df = stock_data_close.iloc[: int(len(stock_data_close) * 0.9) + 1]  # 90%
        test_df = stock_data_close.iloc[int(len(stock_data_close) * 0.9) :]  # 10%

        # Define training model
        model = AutoReg(train_df["Close"], 180).fit(cov_type="HC0")

        # Predict data for test data
        predictions = model.predict(
            start=test_df.index[0], end=test_df.index[-1], dynamic=True
        )

        # Predict 90 days into the future
        forecast = model.predict(
            start=test_df.index[0],
            end=test_df.index[-1] + dt.timedelta(days=10),
            dynamic=True,
        )



        if market_index_select == "Nifty":
            # Check if the data is not None
            if train_df is not None and (forecast >= 0).all() and (predictions >= 0).all():
                # Add a title to the stock prediction graph
                st.markdown("## **Stock Prediction**")

                # Create a plot for the stock prediction
                fig = go.Figure(
                    data=[
                        go.Scatter(
                            x=train_df.index,
                            y=train_df["Close"],
                            name="Train",
                            mode="lines",
                            line=dict(color="blue"),
                        ),
                        go.Scatter(
                            x=test_df.index,
                            y=test_df["Close"],
                            name="Test",
                            mode="lines",
                            line=dict(color="orange"),
                        ),
                        go.Scatter(
                            x=forecast.index,
                            y=forecast,
                            name="Forecast",
                            mode="lines",
                            line=dict(color="red"),
                        ),
                        go.Scatter(
                            x=test_df.index,
                            y=predictions,
                            name="Test Predictions",
                            mode="lines",
                            line=dict(color="green"),
                        ),
                    ]
                )

                # Customize the stock prediction graph
                fig.update_layout(xaxis_rangeslider_visible=False)

                # Use the native streamlit theme.
                st.plotly_chart(fig, use_container_width=True)

        # If the data is None
        else:
            # Add a title to the stock prediction graph
            st.markdown("## **Stock Prediction**")

            # Add a message to the stock prediction graph
            st.markdown("### **No data available for the selected stock**")


pg = st.navigation([
    st.Page(portfolio_analytics, title="Portfolio Analytics", icon=":material/shopping_basket:"),
    st.Page(market_insights, title="Market Analytics", icon=":material/price_change:"),
])
pg.run()





# lc, mc, rc = st.columns(3)
#
#
#
# with st.sidebar:
#   selected_option = option_menu(
#     "TraDatAnalytix",
#     ['Authentication','Global Markets','Indices Data','Open Interest Data', 'FII/DII Data','Pick Outperformers' ,'Trading Strategy'],
#     icons = ['globe', 'globe','body-text','bar-chart-fill', 'gear', 'currency-exchange' ,'option'],
#     menu_icon = "cast",
#     default_index = 0
#   )
