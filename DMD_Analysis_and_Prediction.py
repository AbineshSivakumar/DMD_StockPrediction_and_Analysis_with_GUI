import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import pydmd

def readSymbol(stock_exchange):
    symbol = pd.read_csv(f"DATA/{stock_exchange}.csv")
    symbol = symbol.iloc[:,0].tolist()
    if stock_exchange == "nse100_SYM":
        matx = pd.read_csv("DATA/NSE100.csv",header=None)
    if stock_exchange == "nse50_sym":
        matx = pd.read_csv("DATA/nse50.csv",header=None)

    matx = pd.DataFrame(matx.transpose())
    matx = matx.fillna(0)

    return symbol, matx


def dmd_pred_plot(dmd,diff,d_ind,top):  
    # Find the indices of the top 5 elements
    dmode = dmd.modes[:,d_ind].real
    top_indices = np.argsort(dmode.real)[-top:]
    st.info("RECOMMENDED STOCKS")
    for i in top_indices:
        st.markdown(f"{symbol[i]} [{i}]")
    # Mark the indices on the plot
    plt.plot(diff)
    plt.scatter(top_indices,diff[top_indices], color='red')
    plt.title(" PERFORMACE COMPARISON - RECOMENDED STOCK")
    plt.xlabel("Stocks")
    plt.ylabel("Performance")
    st.pyplot()

def dmd_analysis(result_comparison,j,top=3,i=12) :
    mat = matx.iloc[:, -(i+j):-j]
    mat = mat.to_numpy()

    # dmd 
    dmd = pydmd.DMD(svd_rank=mat.shape[1])
    dmd.fit(mat)
    dmd.dmd_time['tend'] = dmd.dmd_time['tend'] + 9

    # creating the matrix mat 
    # i days are trained j th day from last 

    #setting the test set 
    day0 = matx.iloc[:, -j-1]
    day1 = matx.iloc[:, -j]
    day2 = matx.iloc[:, -j+1]
    day3 = matx.iloc[:, -j+2]
    day4 = matx.iloc[:, -j+3]
    day5 = matx.iloc[:, -j+4]
    day6 = matx.iloc[:, -j+5]
    day7 = matx.iloc[:, -j+6]
    day8 = matx.iloc[:, -j+7]
    day9 = matx.iloc[:, -j+8]


    mat = matx.iloc[:, -(i+j):-j]
    mat=mat.to_numpy()
    day = np.array([day0,day1,day2,day3,day4,day5,day6,day7,day8,day9])
    day_r = day[result_comparison]
    d_ind = np.argmax(np.abs(dmd.amplitudes.real))

    #if any(np.abs(dmd.eigs.real)>1) and dmd.amplitudes.real[d_ind] > 0 :
    if np.abs(dmd.eigs.real[d_ind])>1 and dmd.eigs.imag[d_ind] == 0 and dmd.amplitudes.real[d_ind] > 0 :
        #stylised font for growing
        st.success("Growing 📈")
        st.line_chart(dmd.amplitudes.real)
        #plot axs[1] in the function dmd_pred_plot
        dmd_pred_plot(dmd, day_r-day0, d_ind, top)
    else:
        st.error("Shrinking📉")
        st.write("No growing eigen value found")
        st.line_chart(dmd.amplitudes.real)
        if dmd.amplitudes.real[d_ind] > 0 :
            st.write("Amplitude is positive")
        dmd_pred_plot(dmd, day_r-day0, d_ind, top)
    
    return ""

def dmd_pred(j,i,stock_name):
    # creating the matrix mat 
    # i days are trained j th day from last 
    mat = matx.iloc[:, -(i+j):-j]
    mat = mat.to_numpy()
    # index where the stock is equal to the symbol
    stock_ind = symbol.index(stock_name)
    #setting the test set 
    day0 = matx.iloc[:, -j-1]
    day1 = matx.iloc[:, -j]
    day2 = matx.iloc[:, -j+1]
    day3 = matx.iloc[:, -j+2]
    day4 = matx.iloc[:, -j+3]
    day5 = matx.iloc[:, -j+4]
    day6 = matx.iloc[:, -j+5]
    day7 = matx.iloc[:, -j+6]
    day8 = matx.iloc[:, -j+7]
    day9 = matx.iloc[:, -j+8]

    # dmd 
    dmd = pydmd.DMD(svd_rank=mat.shape[1])
    dmd.fit(mat)
    dmd.dmd_time['tend'] = dmd.dmd_time['tend'] + 9


    # prediction

    pred = dmd.reconstructed_data
    pday1 = np.real(pred[:,-9])
    pday2 = np.real(pred[:,-8])
    pday3 = np.real(pred[:,-7])
    pday4 = np.real(pred[:,-6])
    pday5 = np.real(pred[:,-5])
    pday6 = np.real(pred[:,-4])
    pday7 = np.real(pred[:,-3])
    pday8 = np.real(pred[:,-2])
    pday9 = np.real(pred[:,-1])

    # Using st.write() to write the predictions and error in a tabular format
    st.info(("ERROR & PREDICTIONS"))
    data = {'Day': ['Day1', 'Day2', 'Day3', 'Day4', 'Day5', 'Day6', 'Day7', 'Day8', 'Day9'],
            'Prediction': [pday1[stock_ind], pday2[stock_ind], pday3[stock_ind], pday4[stock_ind], pday5[stock_ind], pday6[stock_ind], pday7[stock_ind], pday8[stock_ind], pday9[stock_ind]],
            'Error': [np.abs(day1[stock_ind]-pday1[stock_ind]), np.abs(day2[stock_ind]-pday2[stock_ind]), np.abs(day3[stock_ind]-pday3[stock_ind]), np.abs(day4[stock_ind]-pday4[stock_ind]), np.abs(day5[stock_ind]-pday5[stock_ind]), np.abs(day6[stock_ind]-pday6[stock_ind]), np.abs(day7[stock_ind]-pday7[stock_ind]), np.abs(day8[stock_ind]-pday8[stock_ind]), np.abs(day9[stock_ind]-pday9[stock_ind])]}

    df = pd.DataFrame(data)
    st.table(df.style.set_table_styles([dict(selector='th', props=[('text-align', 'center')])]))

    return ""


st.title("Stock Price Prediction")

stock_exchange = ["nse50_sym", "nse100_SYM"]    
index = st.selectbox("Select a Stock Exchange", stock_exchange)

symbol, matx = readSymbol(index)
sym = st.selectbox("Choose a Symbol", symbol)

# input_3 = st.date_input("Enter a date:")
# col1, col2 = st.columns(2)
# from_date = col1.date_input("From:")
# to_date = col2.date_input("To:")

training = st.number_input("Training Set (in days)",step=1,min_value=0)
value = st.slider("Result Comparison", min_value=0, max_value=9, value=3, step=1)
top_stocks = st.number_input("How many stocks would you like us to recommend?",step=1,min_value=0)

# pred_options = ["Next 5 days", "Next 10 days"]
# predict = st.selectbox("Predict for",pred_options)

st.set_option('deprecation.showPyplotGlobalUse', False)

if st.button("Predict"):
    st.write(dmd_analysis(value,training,top_stocks))
    st.write(dmd_pred(training,12,sym))




