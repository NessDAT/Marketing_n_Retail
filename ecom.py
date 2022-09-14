import streamlit as st
import requests
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
import pandas as pd
from pycaret.classification import *
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.io as pio
import plotly.graph_objects as go
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re







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
                        "Demand Forecasting",],
                         icons=['journal-bookmark-fill', 'person-circle', 'diagram-3-fill', 'cart-check','chat-square-text', 
                         'graph-up-arrow'],
                         menu_icon="shop", default_index=0,
                         styles={"nav-link": {"--hover-color": "#eee"}}
                        )
    st.markdown("###")
    

    st.markdown("<a href='https://datawaredemo.azurewebsites.net/'>\
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
    col1, col2= st.columns(2)

    with col1:
        st.subheader("Introduction")
        st.markdown("<p style='text-align:justify;font-size:20px;font-family:helvetica;'> \
            Artificial intelligence (AI) and machine learning (ML) significantly impact the\
             retail world, particularly for companies that rely on online sales, where using\
              some kind of AI technology is very common nowadays. Big players like eBay,\
               Amazon or Alibaba have successfully integrated AI across the entire sales\
                cycle, from storage logistics to post-sale customer service.</p>",
                    unsafe_allow_html=True)

        st.markdown("<p style='text-align:justify;font-size:20px;font-family:helvetica;'>\
            From clothes to groceries to household items, the possibilities in the \
            retail space are full of promise. The use cases presented in here \
            are a fraction of the feasible Machine Learning projects and serve as \
            examples of what can be done today in the Retail space. That being said, \
            many companies have very unique needs that could be served with data and \
            custom Machine Learning development.</p>",
            unsafe_allow_html=True)

        st.markdown("##")
        st.markdown("##")

    with col2:
        st.markdown("#")
        st.image("images/girl.png")

    st.markdown("###")

    st.subheader("Use Cases Being Tackled In This Platform")

    st.markdown("<ul>\
            <li style='font-size:20px;font-family:helvetica;'>Ecommerce Churn Prediction</li>\
            <li style='font-size:20px;font-family:helvetica;'>Market Segmentation</li>\
            <li style='font-size:20px;font-family:helvetica;'>Market Basket Analysis</li>\
            <li style='font-size:20px;font-family:helvetica;'>Review Classfication</li>\
            <li style='font-size:20px;font-family:helvetica;'>Demand Forecasting</li>\
        </ul>\
        </p>",
        unsafe_allow_html=True)











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

                loaded_model2 = load_model('Churn/models/Churn_Model2')
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
                loaded_model = load_model("Churn/models/Churn_Model2")
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
if selected == "Market Segmentation":
    expander1 = st.expander('Upload your dataset')
    uploaded_file = expander1.file_uploader(".", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

    if uploaded_file is not None:
        st.write(df)



#-----------------------------------------------------------------------------------------------------------------------
# THIS IS THE MARKET BASKET ANALYSIS SECTION
#------------------------------------------------------------------------------------------------------------------------





#-----------------------------------------------------------------------------------------------------------------------
# THIS IS THE REVIEW CLASSIFICAITION SECTION
#------------------------------------------------------------------------------------------------------------------------
if selected == "Review Classfication":
    choose = option_menu(None,["Appeciation Details","Sentiment Analysis"],
        icons=['person-rolodex', 'grid-3x3-gap-fill'],default_index=0,orientation="horizontal")

    #--------------------------------------------------------------------------------------------------------------------
    # APPECIATION DETAILS
    #--------------------------------------------------------------------------------------------------------------------

    if choose == "Appeciation Details":
        df = pd.read_csv("data/product_data.csv")

        #---------------------------------------------------------------------------------------------------------------
        #  TOP KPIs
        #---------------------------------------------------------------------------------------------------------------

        rev_col,price_col,rating_col,pdts_col = st.columns([0.5,0.7,1,3.7])

        rev_col.markdown("**Reviews**")
        price_col.markdown("**Average Price**")
        rating_col.markdown("**Average Star Rating**")
            
        #--Multiselector-----------------------------------

        select_pdt = pdts_col.multiselect('Products',options=df['product'].unique(),default=['Fire Tablet','Amazon Fire Tv'])
        df_selection = df.query("product == @select_pdt")
        num_pdt_selected = df_selection['product'].nunique()


        #---Number of Reviews-------------------------------
        num_reviews = df_selection['id'].count()
        rev_col.subheader(num_reviews)

        #---Avg Price---------------------------------------
        avg_price = round(df_selection['price'].mean(),2)
        price_col.subheader(f" $ {avg_price}")

        #--Star Ratings ------------------------------------
        avg_rating = round(df_selection['rating'].mean(),1)
        star_rating = ":star:"*int(round(avg_rating,0))
        rating_col.subheader(star_rating)


        st.markdown("____")


        fake_connect = st.button("Connect to Database")
        st.markdown("#")

        ratingXprice,gap,ratingXpdt = st.columns([2,0.1,2])

        #-------------------------------------------------------------------------------------------------------------
        # PIE CHART FOR TOTAL NUMBER OF REVIEWS BY PRODUCT
        #-------------------------------------------------------------------------------------------------------------



        if fake_connect:
            ratingXprice.subheader("Review Distribution by Products")
            ratingXprice.markdown(f"( *Number of Products* : {num_pdt_selected} **SELECTED** )")
            fig_product_reviews = px.pie(
                df_selection,
                names="product",
                hole=0.5,
                color_discrete_sequence=px.colors.sequential.RdBu,
                template= "plotly_white"
            )


            colors = ['orangered']
            fig_product_reviews.update_traces(rotation=115,marker=dict(colors=colors))
            fig_product_reviews.update_layout(width=600,height=550)
            ratingXprice.plotly_chart(fig_product_reviews)

        else:
            st.warning('Empty Dashboard. Please Connect to Database')
            
        

        #-------------------------------------------------------------------------------------------------------------
        # HORIZONTAL BAR CHART FOR AVERAGE RATING BY PRODUCT
        #-------------------------------------------------------------------------------------------------------------

        if fake_connect:
            ratingXpdt.subheader("Average Ratings by Products")
            ratingXpdt.markdown(f"( *Number of Products* : {num_pdt_selected} **SELECTED** )")
            rating_by_pdt = (
                df_selection.groupby(by=["product"]).mean()[['rating']]
            )

            fig_product_ratings = px.bar(
                rating_by_pdt,
                x="rating",
                y=rating_by_pdt.index,
                text_auto=True,
                color_discrete_sequence=["#F35106"]*len(rating_by_pdt),
                template= "plotly_white"
            )

            #fig_product_ratings.update_layout(width=600,height=300,margin=dict(l=1,r=1,b=1,t=1))
            ratingXpdt.plotly_chart(fig_product_ratings)


        #-------------------------------------------------------------------------------------------------------------
        # VERTICAL BAR CHART FOR AVERAGE PRICE BY PRODUCT
        #-------------------------------------------------------------------------------------------------------------


        if fake_connect:
            st.subheader("Distribution of Prices for each Product")
            st.markdown("**Average Price of Products in USD**   ( *Number of Products* : **ALL** )")
            temp_df = df.groupby(by=["product"]).mean()
            fig = px.histogram(temp_df, x=temp_df.index, y='price',text_auto=True,color_discrete_sequence=["#F35106"])
            fig.update_layout(width=1400,height=500,margin=dict(l=1,r=1,b=1,t=1))
            st.plotly_chart(fig)




        



    #--------------------------------------------------------------------------------------------------------------------
    # SENTIMENT ANALYSIS
    #--------------------------------------------------------------------------------------------------------------------

    if choose=="Sentiment Analysis":
        expander1 = st.expander('Upload your input CSV file')
        uploaded_file = expander1.file_uploader(".", type=["csv"])

        st.markdown("#")

        if uploaded_file is not None:
            sent_df = pd.read_csv(uploaded_file)


        rev_class,gap,wordfreq = st.columns([2,0.1,2])

        

        #----------------------------------------------------------------------------------------------------------
        #   REVIEW CLASSIFICATION
        #----------------------------------------------------------------------------------------------------------

        if uploaded_file is not None:
            rev_class.subheader("Review Classfication")
            review_class_fig = px.pie(
                sent_df,
                names="review_sentiment",
                hole=0.5,
                color_discrete_sequence=px.colors.sequential.RdBu,
                template = "plotly_white")

            colors = ['orangered', 'dimgrey']
            review_class_fig.update_layout(width=600,height=550, title_x=0.2)
            review_class_fig.update_traces(rotation=115,marker=dict(colors=colors))

            rev_class.plotly_chart(review_class_fig)
        else:
            st.warning("Empty Dashboard. Please Upload Dataset")



        

        #----------------------------------------------------------------------------------------------------------
        #   WORD FREQUENCY BY SENTIMENT
        #----------------------------------------------------------------------------------------------------------

        if uploaded_file is not None:
            lemma = WordNetLemmatizer()
            stop_words = stopwords.words('english')
            def text_prep(x):
                corp = str(x).lower()
                corp = re.sub('[^a-zA-Z]+',' ',corp).strip()
                tokens = word_tokenize(corp)
                words = [t for t in tokens if t not in stop_words]
                lemmatize = [lemma.lemmatize(w) for w in words]

                return lemmatize

            preprocess_tag = [text_prep(i) for i in sent_df['reviews.text']]
            sent_df['preprocess_txt'] = preprocess_tag

            sent_df['total_len'] = sent_df['preprocess_txt'].map(lambda x: len(x))

            file = open('Review/negative-words.txt', 'r')
            neg_words = file.read().split()
            file = open('Review/positive-words.txt', 'r')
            pos_words = file.read().split()

            num_pos = sent_df['preprocess_txt'].map(lambda x: len([i for i in x if i in pos_words]))
            sent_df['pos_count'] = num_pos
            num_neg = sent_df['preprocess_txt'].map(lambda x: len([i for i in x if i in neg_words]))
            sent_df['neg_count'] = num_neg


        if uploaded_file is not None:
            wordfreq.subheader("Word Frequency in Reviews")
            pdt_select = wordfreq.selectbox('Select Product',options=sent_df['product'].unique())
            pos_neg_selection = sent_df.query("product == @pdt_select")
            pos_neg_df = pos_neg_selection[['product','preprocess_txt','total_len','pos_count','neg_count']]
            pos_neg_df.rename(columns={
                'preprocess_txt':'Words in Review',
                'total_len':'Total number of words in Review',
                'pos_count':'Number of Positive Words',
                'neg_count':'Number of Negative Words'
                },inplace=True
            )
            wordfreq.write(pos_neg_df.head(10))


            temp_df = pos_neg_df.groupby(by=['product']).sum()
            #pos_neg_fig = px.bar(temp_df, x="product", y="pos_count",color="product", barmode="group")

            #pos_count_fig = px.bar(
                #pos_neg,
               # y='pos_count',
                #x=pos_neg.index,
               # title='<b>Positive Word Frequency by Products</b>',
                #color_discrete_sequence=['#F35106']*len(pos_neg),
                #template='plotly_white'
            #)

        pos_neg_sum,gap,revXpdt = st.columns([2,0.1,2])
        if uploaded_file is not None:
            temp_df2 = temp_df[temp_df.index == pdt_select]
            pos_word_sum = temp_df2['Number of Positive Words'].values
            neg_word_sum = temp_df2['Number of Negative Words'].values
            pos_neg_sum.subheader(f"Number of Identified Postive Words {pos_word_sum}")
            pos_neg_sum.subheader(f"Number of Identified Negative Words {neg_word_sum}")


        #----------------------------------------------------------------------------------------------------------
        #   PRODUCT PRICE VS RATING
        #----------------------------------------------------------------------------------------------------------


        if uploaded_file is not None:
            revXpdt.subheader("Rating VS Sentiment Vs Price")
            fig_rating_price = px.scatter(
                sent_df,
                x='rating',
                y='price',
                color='review_sentiment',
                #template= "plotly_white"
                )
            fig_rating_price.update_layout(width=700,height=400,margin=dict(l=1,r=1,b=1,t=1))
            revXpdt.plotly_chart(fig_rating_price)








#-----------------------------------------------------------------------------------------------------------------------
# THIS IS THE LIFETIME VALUE SECTION
#------------------------------------------------------------------------------------------------------------------------


















 
