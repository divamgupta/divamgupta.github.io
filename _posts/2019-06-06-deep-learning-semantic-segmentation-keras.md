---

layout: post

title:  "A Beginner's guide to Deep Learning based Semantic Segmentation using Keras"

author: divam

categories: [ image-segmentation ]

comments: true

hidden: false

image: "assets/images/posts/imgseg/teaser.png"

featured: false

---

Pixel-wise image segmentation is a well-studied problem in computer vision. The task of semantic image segmentation is to classify each pixel in the image. In this post, we will discuss how to use deep convolutional neural networks to do image segmentation. We will also dive into the implementation of the pipeline -- from preparing the data to building the models. 

**I have packaged all the code in an easy to use repository: [https://github.com/divamgupta/image-segmentation-keras](https://github.com/divamgupta/image-segmentation-keras)**

Deep learning and convolutional neural networks (CNN) have been extremely ubiquitous in the field of computer vision. CNNs are popular for several computer vision tasks such as Image Classification, Object Detection, Image Generation, etc. Like for all other computer vision tasks, deep learning has surpassed other approaches for image segmentation


## What is semantic segmentation 

Semantic image segmentation is the task of classifying each pixel in an image from a predefined set of classes. In the following example, different entities are classified.





![]({{ site.baseurl }}/assets/images/posts/imgseg/image15.png?style=centerme)
*Semantic segmentation of a bedroom image*
{: style="font-size: 80%; text-align: center;"}

In the above example, the pixels belonging to the bed are classified in the class “bed”, the pixels corresponding to the walls are labeled as “wall”, etc. 

In particular, our goal is to take an image of size W x H x 3 and generate a W x H matrix containing the predicted class ID’s corresponding to all the pixels. 





![]({{ site.baseurl }}/assets/images/posts/imgseg/image14.png?style=centerme)
*Image source: jeremyjordan.me*
{: style="font-size: 80%; text-align: center;"}

Usually, in an image with various entities, we want to know which pixel belongs to which entity, For example in an outdoor image, we can segment the sky, ground, trees, people, etc. 

Semantic segmentation is different from object detection as it does not predict any bounding boxes around the objects. We do not distinguish between different instances of the same object. For example, there could be multiple cars in the scene and all of them would have the same label. 




