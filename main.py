import uuid
import streamlit as st
from streamlit_option_menu import option_menu
from settings import *
import face_recognition
# st.set_option("deprication.showPyplotGlobalUse",False)
# st.set_option("deprication.showPyplotGlobalUse",False)

st.sidebar.markdown("Made By AYUSH NAUTIYAL :smile:")
# attend_img=""
st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("https://wytech.co.ke/wp-content/uploads/2020/06/School-Attendance-Systems.jpg");
            background-attachment: fixed;
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True)
if st.sidebar.button("Click to clear the Cache"):
    shutil.rmtree(VISITOR_DB,ignore_errors=True)
    os.mkdir(VISITOR_DB)
    shutil.rmtree(VISITOR_HISTORY,ignore_errors=True)
    os.mkdir(VISITOR_HISTORY)
if not os.path.exists(VISITOR_DB):
    os.mkdir(VISITOR_DB)
if not os.path.exists(VISITOR_HISTORY):
    os.mkdir(VISITOR_HISTORY)
    
def main():
    st.sidebar.header("About")
    selected=option_menu(None,["Vistor Validation","View Vistor History","Add to Database"],icons=['camera','clock-history','person-plus'],default_index=0,orientation='horizontal')
    if selected=="Vistor Validation":
        vistor_id=uuid.uuid1()
        img_file=st.camera_input("Take a Picture")
        if img_file is not None:
            bytes_data=img_file.getvalue()
            image_arr=cv2.imdecode(np.frombuffer(bytes_data,np.uint8),cv2.IMREAD_COLOR)
            image_array_copy=cv2.imdecode(np.frombuffer(bytes_data,np.uint8),cv2.IMREAD_COLOR)    
            with open(os.path.join(VISITOR_HISTORY,f'{vistor_id}.jpg'),'wb') as file:
                file.write(img_file.getbuffer())
                st.success("IMAGE SAVED SUCESSFULLY !")
                max_face=0
                rois=[]
                face_locations=face_recognition.face_locations(image_arr)
                encodes_curr_frame=face_recognition.face_encodings(image_arr,face_locations)
                for idx,(top,right,bottom,left) in enumerate(face_locations):
                    rois.append(image_arr[top:bottom,left:right].copy())
                    cv2.rectangle(image_arr,(left,top),(right,bottom),COLOR_DARK,2)
                    cv2.rectangle(image_arr,(left,bottom+35),(right,bottom),COLOR_DARK,cv2.FILLED)
                    font=cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(image_arr,f"#Image No:-{idx}",(left+5,bottom+25),font,.55,COLOR_WHITE,1)
                st.image(BGR_to_RGB(image_arr),width=720)
                max_faces=len(face_locations)
                if max_faces>0:
                    col1,col2=st.columns(2)
                    face_idxs = col1.multiselect("Select face#", range(max_faces),
                                                 default=range(max_faces))
                    similarity_threshold = col2.slider('Select Threshold for Similarity',
                                                         min_value=0.0, max_value=1.0,
                                                         value=0.5)
                                                    ## check for similarity confidence greater than this threshold
                    flag_show = False
                    if ((col1.checkbox('Click to proceed!')) & (len(face_idxs)>0)):
                        dataframe_new = pd.DataFrame()
                        ## Iterating faces one by one
                        for face_idx in face_idxs:
                            ## Getting Region of Interest for that Face
                            roi = rois[face_idx]
                            # st.image(BGR_to_RGB(roi), width=min(roi.shape[0], 300))
                            # initial database for known faces
                            database_data = initialize_data()
                            # st.write(DB)
                            ## Getting Available information from Database
                            face_encodings  = database_data[COLS_ENCODE].values
                            dataframe       = database_data[COLS_INFO]
                            # Comparing ROI to the faces available in database and finding distances and similarities
                            faces = face_recognition.face_encodings(roi)
                            # st.write(faces)
                            if len(faces) < 1:
                                ## Face could not be processed
                                st.error(f'Please Try Again for face#{face_idx}!')
                            else:
                                face_to_compare = faces[0]
                                ## Comparing Face with available information from database
                                dataframe['distance'] = face_recognition.face_distance(face_encodings, face_to_compare)
                                dataframe['distance'] = face_recognition.face_distance(face_encodings,
                                                                                       face_to_compare)
                                dataframe['distance'] = dataframe['distance'].astype(float)
                                dataframe['similarity'] = dataframe.distance.apply(
                                    lambda distance: f"{face_distance_to_conf(distance):0.2}")
                                dataframe['similarity'] = dataframe['similarity'].astype(float)
                                dataframe_new = dataframe.drop_duplicates(keep='first')
                                dataframe_new.reset_index(drop=True, inplace=True)
                                dataframe_new.sort_values(by="similarity", ascending=True)
                                dataframe_new = dataframe_new[dataframe_new['similarity'] > similarity_threshold].head(1)
                                dataframe_new.reset_index(drop=True, inplace=True)
                                if dataframe_new.shape[0]>0:
                                    (top, right, bottom, left) = (face_locations[face_idx])
                                    ## Save Face Region of Interest information to the list
                                    rois.append(image_array_copy[top:bottom, left:right].copy())
                                    # Draw a Rectangle Red box around the face and label it
                                    cv2.rectangle(image_array_copy, (left, top), (right, bottom), COLOR_DARK, 2)
                                    cv2.rectangle(image_array_copy, (left, bottom + 35), (right, bottom), COLOR_DARK, cv2.FILLED)
                                    font = cv2.FONT_HERSHEY_DUPLEX
                                    cv2.putText(image_array_copy, f"#{dataframe_new.loc[0, 'Name']}", (left + 5, bottom + 25), font, .55, COLOR_WHITE, 1)
                                    ## Getting Name of Visitor
                                    name_visitor = dataframe_new.loc[0, 'Name']
                                    attendance(vistor_id, name_visitor)
                                    flag_show = True
                                else:
                                    st.error(f'No Match Found for the given Similarity Threshold! for face#{face_idx}')
                                    st.info('Please Update the database for a new person or click again!')
                                    attendance(vistor_id, 'Unknown')
                        if flag_show == True:
                            st.image(BGR_to_RGB(image_array_copy), width=720)



                else:
                    st.error('No human face detected.')
    if selected == 'View Vistor History':
        view_attendace()
    if selected == 'Add to Database':
        col1, col2, col3 = st.columns(3)
        face_name  = col1.text_input('Name:', '')
        pic_option = col2.radio('Upload Picture',
                                options=["Upload a Picture",
                                         "Click a picture"])
        if pic_option == 'Upload a Picture':
            img_file_buffer = col3.file_uploader('Upload a Picture',
                                                 type=allowed_image_type)
            if img_file_buffer is not None:
                file_bytes = np.asarray(bytearray(img_file_buffer.read()),
                                        dtype=np.uint8)
        elif pic_option == 'Click a picture':
            img_file_buffer = col3.camera_input("Click a picture")
            if img_file_buffer is not None:
                file_bytes = np.frombuffer(img_file_buffer.getvalue(),
                                           np.uint8)
                if ((img_file_buffer is not None) & (len(face_name) > 1) &
                st.button('Click to Save!')):
                    image_arr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    with open(os.path.join(VISITOR_DB,
                                   f'{face_name}.jpg'), 'wb') as file:
                        file.write(img_file_buffer.getbuffer())
                # st.success('Image Saved Successfully!')

                    face_locations = face_recognition.face_locations(image_arr)
                    encodesCurFrame = face_recognition.face_encodings(image_arr,
                                                              face_locations)
                    df_new = pd.DataFrame(data=encodesCurFrame,
                                  columns=COLS_ENCODE)
                    df_new[COLS_INFO] = face_name
                    df_new = df_new[COLS_INFO + COLS_ENCODE].copy()

            # st.write(df_new)
            # initial database for known faces
                    DB = initialize_data()
                    add_data_db(df_new)
if __name__ == "__main__":
    main()