# 17/07/18 Daily Report


## Machine Learning on Android

### Necessity of ML on Android
1. UX
  Regarding response speed of service and case of offline situation, on-device machine learning is necessary.

2. Cost
  - Battery consumption cost
  - Data network consumption ; whenever request on server and case of uploading large size of data. 

3. Privacy
  Whenever user does not want to provide own data to ML platform.
  
  
### Ways of ML on Android
There are three ways to enable machine learning on Android platform.

#### Using JNI to bridge into the NDK

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/JNI_NDK.png" width="700" height="500">

  * NDK : Native Development Kit
    NDK connects App(Java) and Library(C/C++) using JNI interface. We say "native method" for implementation using C/C++ in JVM.
    JDK supports JNI, so calling C/C++ code in JVM is possible.

This code is complex and hard to maintain, e.g. the JNI code needs to be built differently from normal Android Studio/Gradle builds.


#### TensorFlowInferenceInterface class

To make this easier, in late 2016 Google added the TensorFlowInferenceInterface class (GitHub commits). 
This helped standardize how to interface with TensorFlow models from Java. 
It provides these prebuilt libraries:

  - libandroid_tensorflow_inference_java.jar — the Java interface layer.
  - libtensorflow_inference.so — the JNI code that talks to the TensorFlow model.
  
The picture below is screenshot of one example. 
  
<img src="https://github.com/jwcse/DeepLearning/blob/master/img/TensorFlowInferenceInterface_ex.png" width="600" height="400">



#### More Simple Using TensorFlowInferenceInterface

Simply by adding a dependency to your build.gradle and then using the TensorFlowInferenceInterface class, it has been much easier.

What we have to do is:
  1.Include the compile 'org.tensorflow:tensorflow-android:+' dependency in your build.gradle.
  
  2. Use the TensorFlowInferenceInterface to interface with your model.


### Efficient TensorFlow model for Mobile : TensorFlow Lite

TensorFlow Lite is TensorFlow’s lightweight solution for mobile and embedded devices. It lets you run machine-learned models on mobile devices with low latency, so you can take advantage of them to do classification, regression or anything else you might want without necessarily incurring a round trip to a server.

TensorFlow Lite is comprised of a runtime on which you can run pre-existing models, and a suite of tools that you can use to prepare your models for use on mobile and embedded devices.

It’s not yet designed for training models. Instead, you train a model on a higher powered machine, and then convert that model to the .TFLITE format, from which it is loaded into a mobile interpreter.


<img src="https://github.com/jwcse/DeepLearning/blob/master/img/tensorflowlite.png" width="450" height="406">
