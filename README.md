# face_detection

แนวคิดของ project นี้คือการใช้โมเดล Haar Cascade Classifier ใน OpenCV เพื่อตรวจจับใบหน้า โปรแกรมจะอ่านและแปลงภาพเป็นขาวดำ จากนั้นทำการตรวจจับใบหน้าในภาพและวาดกรอบสีฟ้ารอบบริเวณที่ตรวจจับได้ ผลลัพธ์จะแสดงภาพที่มีการตรวจจับและจำนวนบุคคลในภาพบนหน้าต่างโปรแกรม ดังนั้นโปรแกรมนี้จึงเป็นเพียงตัวอย่างเบื้องต้นที่สามารถใช้เพื่อเรียนรู้การใช้ Haar Cascade Classifier ใน OpenCV สำหรับตรวจจับวัตถุ

![face_detection](https://github.com/srpp0717/face_detection/assets/148683906/40674907-c4ef-44ab-8a97-4fa902f822e8)


### **ขั้นตอนการทำงานของโปรแกรม**

**1. โหลดรูปภาพและโมเดล Haar Cascade Classifier**

โปรแกรมนี้ใช้ cv.CascadeClassifier เพื่อโหลดโมเดล Haar Cascade Classifier ซึ่งเป็นอัลกอริทึมที่ใช้ในการตรวจจับวัตถุ โดยได้รับการฝึกสอนให้รู้จักลักษณะของวัตถุที่ต้องการตรวจจับ สำหรับตรวจจับใบหน้า Classifier นี้ถูกฝึกสอนให้รู้จักลักษณะของใบหน้าจากไฟล์ face-detect-model.xml

**2. อ่านภาพและแปลงภาพ**

โปรแกรมจะอ่านภาพที่ระบุใน image_path และ cv.cvtColor ใช้ในการแปลงภาพจากรูปสีไปยังรูปขาวดำ (grayscale) เพื่อลดขนาดข้อมูลและเพิ่มประสิทธิภาพในการประมวลผล ปรับขนาดของรูปภาพให้พอดีกับขนาดหน้าต่างที่สามารถแสดงได้ โดยให้ความกว้างของภาพไม่เกิน 800 พิกเซล

**3. การตรวจจับใบหน้า**

face_model.detectMultiScale ใช้ในการตรวจจับใบหน้าในภาพ และทำการปรับพารามิเตอร์ scaleFactor, minNeighbors, และ minSize เพื่อปรับปรุงความแม่นยำ 

**4. วาดกรอบบนใบหน้า**

หากมีการตรวจจับใบหน้าสำเร็จ โปรแกรมจะวาดกรอบสีฟ้ารอบบริเวณที่ตรวจจับได้โดยใช้ cv.rectangle และนับจำนวนใบหน้าที่ตรวจจับได้

![result_1](https://github.com/srpp0717/face_detection/assets/148683906/28be74dd-a234-4aa9-9523-62ae98786451)


**5. แสดงผลลัพธ์**

cv.imshow ใช้ในการแสดงภาพที่ผ่านการตรวจจับ โดยจะมีกรอบสีฟ้าบริเวณใบหน้าของบุคคลที่ตรวจจับได้ ใช้ cv.putText เพื่อแสดงจำนวนใบหน้าที่ตรวจจับได้บนภาพที่ปรับขนาดแล้ว และโปรแกรมจะรอจนกว่าผู้ใช้จะปิดหน้าต่าง

![result_2](https://github.com/srpp0717/face_detection/assets/148683906/22c98f24-4113-4096-a493-b3bc80a276d1)


**6. จบโปรแกรม**

cv2.waitKey(0) และ cv.destroyAllWindows() ใช้ในการรอรับคีย์บอร์ดจากผู้ใช้เพื่อปิดหน้าต่าง

