import face_recognition as fr
import cv2 as cv

known_img = fr.load_image_file(r"F:\Python\Vs code programming files\Imgs2\Pratham\WhatsApp Image 2024-01-30 at 00.20.28.jpeg")
encoding_known_img = fr.face_encodings(known_img) 
print(encoding_known_img) 
# cv.imshow("Known Image", encoding_known_img)

video = cv.VideoCapture(0)

isTrue = True

while isTrue:
    isTrue,frame = video.read()

    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    face_boundries = fr.face_locations(frame)
    
    encoding_unknown_face = fr.face_encodings(frame)
    #print(encoding_unknown_face)

    try:

        for faces in encoding_unknown_face:
            compare = fr.compare_faces([encoding_known_img], faces)
            print("Compare:", compare)
            [true] = compare
            print(face_boundries)

            cv.cvtColor(frame, cv.COLOR_RGB2BGR)

            if true.all():
                for (y,x,h,w) in face_boundries:
                    cv.rectangle(frame,(w,y),(x,h),(0,0,255), 1)
                    #cv.putText(frame, "Pratham", (w,y-20), 1, 1.0, (0,0,255),1)
                    cv.putText(frame, "Pratham", (w + 6, h - 6), cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                
        
            cv.imshow("Face", frame)
    except Exception as e:
        
        print(f" Eception:-  {str(e)}")

    if cv.waitKey(1) & 0XFF == ord(' '):
        break

video.release()
cv.destroyAllWindows()
