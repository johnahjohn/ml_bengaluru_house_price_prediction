
from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
model = load_model('final_project_bengaluru')






def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():
    from PIL import Image
    image = Image.open('banglore_1.jpg')
    image_office = Image.open('banglore_2.jpg')
    st.image(image,use_column_width=True)
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))
    st.sidebar.info('This app is created to predict the house prices at various locations in Bengaluru')
    st.sidebar.success('https://www.pycaret.org')
    st.sidebar.image(image_office)
    st.title("Predicting House Prices")
    if add_selectbox == 'Online':
        location=st.selectbox('location', ['Electronic City Phase II', 'Chikka Tirupathi', 'Uttarahalli',
       'Lingadheeranahalli', 'Kothanur', 'Whitefield', 'Old Airport Road',
       'Rajaji Nagar', 'Marathahalli', 'other', '7th Phase JP Nagar',
       'Gottigere', 'Sarjapur', 'Mysore Road', 'Bisuvanahalli',
       'Raja Rajeshwari Nagar', 'Kengeri', 'Binny Pete', 'Thanisandra',
       'Bellandur', 'Electronic City', 'Ramagondanahalli', 'Yelahanka',
       'Hebbal', 'Kasturi Nagar', 'Kanakpura Road',
       'Electronics City Phase 1', 'Kundalahalli', 'Chikkalasandra',
       'Murugeshpalya', 'Sarjapur  Road', 'HSR Layout', 'Doddathoguru',
       'KR Puram', 'Bhoganhalli', 'Lakshminarayana Pura', 'Begur Road',
       'Devanahalli', 'Varthur', 'Bommanahalli', 'Gunjur', 'Hegde Nagar',
       'Haralur Road', 'Hennur Road', 'Kothannur', 'Kalena Agrahara',
       'Kaval Byrasandra', 'ISRO Layout', 'Garudachar Palya', 'EPIP Zone',
       'Dasanapura', 'Kasavanhalli', 'Sanjay nagar', 'Domlur',
       'Sarjapura - Attibele Road', 'Yeshwanthpur', 'Chandapura',
       'Nagarbhavi', 'Ramamurthy Nagar', 'Malleshwaram', 'Akshaya Nagar',
       'Shampura', 'Kadugodi', 'LB Shastri Nagar', 'Hormavu',
       'Vishwapriya Layout', 'Kudlu Gate', '8th Phase JP Nagar',
       'Bommasandra Industrial Area', 'Anandapura',
       'Vishveshwarya Layout', 'Kengeri Satellite Town', 'Kannamangala',
       ' Devarachikkanahalli', 'Hulimavu', 'Mahalakshmi Layout',
       'Hosa Road', 'Attibele', 'CV Raman Nagar', 'Kumaraswami Layout',
       'Nagavara', 'Hebbal Kempapura', 'Vijayanagar', 'Nagasandra',
       'Kogilu', 'Panathur', 'Padmanabhanagar', '1st Block Jayanagar',
       'Kammasandra', 'Dasarahalli', 'Magadi Road', 'Koramangala',
       'Dommasandra', 'Budigere', 'Kalyan nagar', 'OMBR Layout',
       'Horamavu Agara', 'Ambedkar Nagar', 'Talaghattapura', 'Balagere',
       'Jigani', 'Gollarapalya Hosahalli', 'Old Madras Road',
       'Kaggadasapura', '9th Phase JP Nagar', 'Jakkur', 'TC Palaya',
       'Giri Nagar', 'Singasandra', 'AECS Layout', 'Mallasandra', 'Begur',
       'JP Nagar', 'Malleshpalya', 'Munnekollal', 'Kaggalipura',
       '6th Phase JP Nagar', 'Ulsoor', 'Thigalarapalya',
       'Somasundara Palya', 'Basaveshwara Nagar', 'Bommasandra',
       'Ardendale', 'Harlur', 'Kodihalli', 'Bannerghatta Road', 'Hennur',
       '5th Phase JP Nagar', 'Kodigehaali', 'Billekahalli', 'Jalahalli',
       'Mahadevpura', 'Anekal', 'Sompura', 'Dodda Nekkundi', 'Hosur Road',
       'Battarahalli', 'Sultan Palaya', 'Ambalipura', 'Hoodi',
       'Brookefield', 'Yelenahalli', 'Vittasandra',
       '2nd Stage Nagarbhavi', 'Vidyaranyapura', 'Amruthahalli',
       'Kodigehalli', 'Subramanyapura', 'Basavangudi', 'Kenchenahalli',
       'Banjara Layout', 'Kereguddadahalli', 'Kambipura',
       'Banashankari Stage III', 'Sector 7 HSR Layout', 'Rajiv Nagar',
       'Arekere', 'Mico Layout', 'Kammanahalli', 'Banashankari',
       'Chikkabanavar', 'HRBR Layout', 'Nehru Nagar', 'Kanakapura',
       'Konanakunte', 'Margondanahalli', 'R.T. Nagar', 'Tumkur Road',
       'GM Palaya', 'Jalahalli East', 'Hosakerehalli', 'Indira Nagar',
       'Kodichikkanahalli', 'Varthur Road', 'Anjanapura', 'Abbigere',
       'Tindlu', 'Gubbalala', 'Cunningham Road', 'Kudlu',
       'Banashankari Stage VI', 'Cox Town', 'Kathriguppe', 'HBR Layout',
       'Yelahanka New Town', 'Sahakara Nagar', 'Rachenahalli',
       'Yelachenahalli', 'Green Glen Layout', 'Thubarahalli',
       'Horamavu Banaswadi', '1st Phase JP Nagar', 'NGR Layout',
       'Seegehalli', 'NRI Layout', 'ITPL', 'Babusapalaya',
       'Iblur Village', 'Ananth Nagar', 'Channasandra', 'Choodasandra',
       'Kaikondrahalli', 'Neeladri Nagar', 'Frazer Town', 'Cooke Town',
       'Doddakallasandra', 'Chamrajpet', 'Rayasandra',
       '5th Block Hbr Layout', 'Pai Layout', 'Banashankari Stage V',
       'Sonnenahalli', 'Benson Town', 'Poorna Pragna Layout',
       'Judicial Layout', 'Banashankari Stage II', 'Karuna Nagar',
       'Bannerghatta', 'Bommenahalli', 'Laggere', 'Prithvi Layout',
       'Banaswadi', 'Sector 2 HSR Layout', 'Shivaji Nagar',
       'Nagavarapalya', 'BTM Layout', 'BTM 2nd Stage', 'Hoskote',
       'Doddaballapur', 'Sarakki Nagar', 'Bharathi Nagar',
       'HAL 2nd Stage', 'Kadubeesanahalli'])
        bhk=st.number_input('bhk' , min_value=1, max_value=6, value=1)
        total_sqft =st.number_input('total_sqft',min_value=300, max_value=7000, value=300)
        bath = st.number_input('bath', min_value=1, max_value=6, value=1)
        output=""
        input_dict={'location':location,'bhk':bhk,'total_sqft':total_sqft,'bath':bath}
        input_df = pd.DataFrame([input_dict])
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = str(output)
        st.success('Price in Lakhs is {}'.format(output))
    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)
def main():
    run()

if __name__ == "__main__":
  main()
