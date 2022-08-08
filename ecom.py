import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
import pandas as pd
from pycaret.classification import *
import plotly.express as px



st.set_page_config(page_title='Retail, eCommernce & Marketing Demos', layout='wide')

# Stylesheet link for bootstap icons
st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.3/font/bootstrap-icons.css">',
    unsafe_allow_html=True)

outter_css = """
<style>
.button{
    --hover-color:#eee
}
</style>
"""

#st.markdown(outter_css,unsafe_allow_html=True)

# Creating the sidebar menu-------------------------------------------------------------------------------------------------------------
with st.sidebar:
    selected = option_menu("Marketing & Retail", ["About", "Churn Prediction","Market Segmentation", "Market Basket Analysis", "Review Classfication", 
                        "Lifetime Value",],
                         icons=['journal-bookmark-fill', 'person-circle', 'diagram-3-fill', 'cart-check','chat-square-text', 
                         'graph-up-arrow'],
                         menu_icon="shop", default_index=0,
                         styles={"nav-link": {"--hover-color": "#eee"}}
                        )
    st.markdown("###")
    

    st.markdown("<a href='https://vanessaattafynn-demo-demo-site-at0bqf.streamlitapp.com/'>\
            <button class='b1' style='background-color:#F35106;color:white; border:None;border-radius:10px;\
            padding:15px;min-height:15px;min-width: 80px;' type='button'>\
            Go Home  <i class='bi bi-box-arrow-up-right'></i></button></a>",unsafe_allow_html=True)

#Get feature importance---------------------------------------------------------------------------------------------------------------
def get_final_column_names(pycaret_pipeline, sample_df):
    for (name, method) in pycaret_pipeline.named_steps.items():
        if method != 'passthrough' and name != 'trained_model':
            print(f'Running {name}')
            sample_df = method.transform(sample_df)
    return sample_df.columns.tolist()

def get_feature_importances_df(pycaret_pipeline, sample_df, n = 10):
    
    final_cols = get_final_column_names(pycaret_pipeline, sample_df)
    
    try:
        variables = pycaret_pipeline["trained_model"].feature_importances_
        
    except:
        variables = np.mean([
                        tree.feature_importances_ for tree in pycaret_pipeline["trained_model"].estimators_
                        ], axis=0)
    
    coef_df = pd.DataFrame({"Variable": final_cols, "Value": variables})
    sorted_df = (
        coef_df.sort_values(by="Value", ascending=False)
        .head(n)
        .sort_values(by="Value", ascending=True).reset_index(drop=True)
    )
    return sorted_df
#-----------------------------------------------------------------------------------------------------------------------
# THIS IS THE ABOUT SECTION
#------------------------------------------------------------------------------------------------------------------------
if selected == "About":
    st.subheader("About Page")



#-----------------------------------------------------------------------------------------------------------------------
# THIS IS THE E-COMMERCE CHURN PREDICTION SECTION
#------------------------------------------------------------------------------------------------------------------------

if selected == "Churn Prediction":
    choose = option_menu(None,["Individual Prediction","Batch Prediction"],
        icons=['person-rolodex', 'grid-3x3-gap-fill'],default_index=0,orientation="horizontal")

# INDIVIDUAL CHURN PREDICTION
#------------------------------------------------------------------------------------------------------------------------
    if choose == "Individual Prediction":
        ID = st.text_input('Customer ID','5000001',max_chars=10)
        def expander_obj():
            with st.expander('Input Customer Data',expanded=True):
                #form = st.form(key='my_form', clear_on_submit=True)
                tenure = st.number_input(label='Length of time since the beginning of the customer relationship (months)', min_value=0, format="%d")
                orderCount = st.number_input(label='Total number of orders places in the Last month', min_value=0, format="%d")
                hoursOnApp = st.number_input(label='Number of hours spent on App', min_value=0, format="%d")
                complain = st.number_input(label='Number of complaints over the last month', min_value=0, format="%d")
                cashback = st.number_input(label='Average Cashback in  the last month', min_value=0, format="%d")
                couponUsed = st.number_input(label='Number of coupons used in  the last month', min_value=0, format="%d")
                daySinceLastOrder = st.number_input(label='Number of Days Since Last Order', min_value=0, format="%d")
                hike = st.number_input(label='Order amount Hike from last year (%)', min_value=0, format="%d")
                warehouse = st.number_input(label='Average Distance from Warehouse to Home (km)', min_value=0, format="%d")
                paymentMode = st.selectbox('Preferred Payment Method', ['Debit Card','Credit Card','Cash on Delivery','E wallet','Unified Payment Interface(UPI)'])
                gender = st.selectbox('Gender', ['Male','Female'])
                maritalStatus = st.selectbox('Marital Status', ['Single','Married','Divorced'])

                st.markdown('##')

            submit_button = st.button('Submit')

            if submit_button:
                st.markdown('##')
                st.markdown('##')
                data = {'Tenure':[tenure],'OrderCount':[orderCount], 'HourSpendOnApp':[hoursOnApp], 'Complain':[complain],
                'CashbackAmount':[cashback], 'CouponUsed':[couponUsed], 'DaySinceLastOrder':[daySinceLastOrder],
                'OrderAmountHikeFromlastYear':[hike], 'WarehouseToHome':[warehouse],'PreferredPaymentMode':[paymentMode],
                'Gender':[gender],'MaritalStatus':[maritalStatus]}
                data = pd.DataFrame.from_dict(data)

                loaded_model2 = load_model('data/Churn_Model2')
                pred_data = predict_model(loaded_model2, data=data)
                pred_data.rename(columns={"Label":"PredictedChurn","Score":"PredictionConfidence"},inplace=True)

                churn_value = pred_data.at[0,'PredictedChurn']
                churn_prob = round((pred_data.at[0, 'PredictionConfidence']*100), 2)
                churn_prob_str = churn_prob.astype(str)

                col1,col2,col3 = st.columns([1,0.7,3])
                with col1:
                    st.markdown("<p style='font-size:1.8em;font-weight:bold;color:#F35106'>Customer ID :</p>",unsafe_allow_html=True)
                    st.markdown("<p style='font-size:1.8em;font-weight:bold;color:#F35106'>Churn : </p>",unsafe_allow_html=True)
                    st.markdown("<p style='font-size:1.8em;font-weight:bold;color:#F35106'>Probability : </p>",unsafe_allow_html=True)
                with col2:
                    st.subheader(ID)
                    if churn_value == 0:
                        st.subheader("NO")
                    else:
                        st.subheader("YES")

                    st.subheader(churn_prob_str + " %")

                

                st.write(pred_data)

        expander_obj()



# BATCH CHURN PREDICTION
#------------------------------------------------------------------------------------------------------------------------

    if choose == "Batch Prediction":
        st.markdown("##")
        uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
        st.markdown("##")
        df = {}
        pred_df = {}
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            with st.expander('Preview Uploaded Data',expanded=True):
                st.write(df)

            if st.button('Submit'):
                st.markdown("____")
                st.subheader("E-commerce Churn Prediction")
                loaded_model = load_model("data/Churn_Model2")
                pred_df = predict_model(loaded_model, data=df)
                pred_df.rename(columns={"Label":"PredictedChurn","Score":"PredictionConfidence"},inplace=True)
                pred_df['Target'] = pred_df["PredictedChurn"]
                pred_df['Target'].replace(0,'Not Churn',inplace=True)
                pred_df['Target'].replace(1,'Churn',inplace=True)
                st.write(pred_df)


# ADDING VISUALIZTION
#------------------------------------------------------------------------------------------------------------------------

                st.markdown("##")
                st.markdown("_____")
                st.subheader("Visualize Results")
                st.markdown("##")
                st.markdown("##")

                col1,col2 = st.columns(2)
                with col1:
                    st.markdown("**Chart of Churn vs Not Churn Customers**")

                    fig1 = px.pie(pred_df, names='Target',hole=0.5,color_discrete_sequence=px.colors.sequential.Jet)
                    fig1.update_layout(width=400,height=500,margin=dict(l=1,r=1,b=1,t=1))
                    st.write(fig1)

                with col2:
                    st.markdown("**Features that Contributed the Most to Prediction**")
                    feature_imps_df = get_feature_importances_df(loaded_model, pred_df, n = 10)
                    feature_imps_df

                with st.expander('See More Info on Churn Customers'):
                    st.markdown("more info")
        


#-----------------------------------------------------------------------------------------------------------------------
# THIS IS THE MARKET SEGMENTATION SECTION
#------------------------------------------------------------------------------------------------------------------------




#-----------------------------------------------------------------------------------------------------------------------
# THIS IS THE MARKET BASKET ANALYSIS SECTION
#------------------------------------------------------------------------------------------------------------------------





#-----------------------------------------------------------------------------------------------------------------------
# THIS IS THE REVIEW CLASSIFICAITION SECTION
#------------------------------------------------------------------------------------------------------------------------






#-----------------------------------------------------------------------------------------------------------------------
# THIS IS THE LIFETIME VALUE SECTION
#------------------------------------------------------------------------------------------------------------------------


















 
