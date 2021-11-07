# VQA_infer




* Download test2015 images
wget -P datasets/coco_extract http://images.cocodataset.org/zips/test2015.zip
unzip datasets/coco_extract/test2015.zip -d datasets/coco_extract/test2015_image
rm datasets/coco_extract/test2015.zip

* Download test2015 feature embedding 
https://awma1-my.sharepoint.com/personal/yuz_l0_tn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyuz%5Fl0%5Ftn%2FDocuments%2Fshare%2Fmscoco%5Fbottom%5Fup%5Ffeatures

* Dowload model to ailibs_data/modular_coattention

https://drive.google.com/drive/folders/1p9OwJWTlTxW4lDE6SVYr4qXj0GklwKPS?usp=sharing

* Download to datasets/vqa/
https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip
https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
https://awma1-my.sharepoint.com/personal/yuz_l0_tn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyuz%5Fl0%5Ftn%2FDocuments%2Fshare%2Fvisualgenome%5Fqa