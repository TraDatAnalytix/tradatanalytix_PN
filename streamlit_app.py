import streamlit as st
import webbrowser
from breeze_connect import BreezeConnect
import datetime
import pandas as pd
from datetime import date, time, datetime, timedelta
import numpy as np
import time
from PIL import Image
from streamlit_option_menu import option_menu
#####

# Site Icon
logo_top = Image.open("./tradatanalytix logo.png")
st.set_page_config(page_title = 'TraDatAnalytix',layout='wide', page_icon=logo_top)




api_key="16G_Mh68829o5105pg1646!O09d2fm43"
#response = 'https://api.icicidirect.com/apiuser/login?api_key='+str(api_key)
#webbrowser.open(response,new=1)

#session_key = st.number_input(label = "Enter Credential", format="%0f")
session_key = 44494017
      # Initialize SDK
breeze = BreezeConnect(api_key="16G_Mh68829o5105pg1646!O09d2fm43")
      # Generate Session
breeze.generate_session(api_secret="6759%V7C09Acs(3567164*J00x@06`)3",
                            session_token=session_key)





####### ICICI Direct Breeze API connection
#43768426

##### Connection Ends ##############



def page1():
    st.title("Portfolio Analytics")


    with st.sidebar:
    # Create a file uploader widget
      uploaded_file = st.file_uploader("Upload your portfolio holdings CSV file", type=["csv"])
      selected_option = option_menu(
        "Portfolio Insights",
        ['Authentication','Global Markets','Indices Data','Open Interest Data', 'FII/DII Data','Pick Outperformers' ,'Trading Strategy'],
        icons = ['globe', 'globe','body-text','bar-chart-fill', 'gear', 'currency-exchange' ,'option'],
        menu_icon = "cast",
        default_index = 0
      )


    if selected_option == 'Indices Data':

        # Check if a file has been uploaded
        if uploaded_file is not None:
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(uploaded_file)

            # Display the DataFrame
            st.write(df)




def page2():
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
        selected2 = option_menu("Post Budget Trading Strategy for:", ["Nifty", "Bank Nifty"],
        icons=['collection', 'bank2'],
        menu_icon="graph-up-arrow", default_index=0, orientation = "horizontal")



        #selected2

        # Test Nifty Dataframe Prices

        sym11 = "DIVLAB"
        sym12 = "NIFTY"


        nifty = []
        ns = [sym12]

        now = datetime.now() - timedelta(days = 1)
        from_date = datetime.now() - timedelta(days = 1825)

        df = breeze.get_historical_data(interval="1day",
                                    from_date= from_date.strftime('2020-04-01T09:20:00.000Z'),
                                    to_date= now.strftime('%Y-%m-%dT15:25:00.000Z'),
                                    stock_code=sym12,
                                    exchange_code="NSE",
                                    product_type="cash")

        data = pd.DataFrame(df['Success'])

        # Convert with a specific format
        data['date_column'] = pd.to_datetime(data['datetime'])
        data.set_index('date_column', inplace=True)
        data2 = data["close"].astype(float)
        data2.info()
        print(data2)


        if not data2.empty:
            nifty.append(data2)


        nifty_prices = pd.concat(nifty, axis = 1)

        nifty_prices.columns = ns

        st.write('Below is a DataFrame:', nifty_prices, 'Above is a dataframe.')


        nifty_prices['returns'] = nifty_prices['NIFTY'].pct_change()
        nifty_prices.index = pd.to_datetime(nifty_prices.index)

        returns = nifty_prices['returns'].dropna()

        # Check the type
        print(type(returns))


pg = st.navigation([
    st.Page(page1, title="Portfolio Analytics", icon=":material/shopping_basket:"),
    st.Page(page2, title="Market Analytics", icon=":material/price_change:"),
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
