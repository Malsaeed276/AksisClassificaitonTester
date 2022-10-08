package com.example.cameraclassificationtester.image_ckassification

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.*
import android.media.ThumbnailUtils
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.example.cameraclassificationtester.RecognitionListener
import com.example.cameraclassificationtester.ml.DocModel
import com.example.cameraclassificationtester.util.YuvToRgbConverter
import com.example.cameraclassificationtester.viewmodel.Recognition
import org.tensorflow.lite.DataType
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.support.model.Model
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder


private const val MAX_RESULT_DISPLAY = 3

class ImageClassification(
    ctx: Context,
    private val listener: RecognitionListener
) :
    ImageAnalysis.Analyzer {
    companion object {
        const val TAG = "TFLite - ODT"
        const val REQUEST_IMAGE_CAPTURE: Int = 1
        private const val MAX_FONT_SIZE = 96F
    }

    // TODO 1: Add class variable TensorFlow Lite Model
    // Initializing the flowerModel by lazy so that it runs in the same thread when the process
    // method is called.
    private val flowerModel: DocModel by lazy {

        // TODO 6. Optional GPU acceleration
        val compatList = CompatibilityList()

        val options = if (compatList.isDelegateSupportedOnThisDevice) {
            Log.d(TAG, "This device is GPU Compatible ")
            Model.Options.Builder().setDevice(Model.Device.GPU).build()
        } else {
            Log.d(TAG, "This device is GPU Incompatible ")
            Model.Options.Builder().setNumThreads(4).build()
        }

        // Initialize the Flower Model
        DocModel.newInstance(ctx, options)
    }

    override fun analyze(imageProxy: ImageProxy) {
        val imageSize = 224
        val classes = arrayOf("report", "document", "non")
        val items = mutableListOf<Recognition>()

        val imageBit = toBitmap(imageProxy)?.let { getImage(it) }
        // TODO 2: Convert Image to Bitmap then to TensorImage
        //val tfImage = TensorImage.fromBitmap(imageBit)

        // Creates inputs for reference.
        val inputFeature0: TensorBuffer =
            TensorBuffer.createFixedSize(intArrayOf(1, imageSize, imageSize, 3), DataType.FLOAT32)

        val byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3)
        byteBuffer.order(ByteOrder.nativeOrder())


        // get 1D array of 224 * 224 pixels in image
        val intValues = IntArray(imageSize * imageSize)
        imageBit?.getPixels(intValues, 0, imageBit.width, 0, 0, imageBit.width, imageBit.height)

        // iterate over pixels and extract R, G, and B values. Add to bytebuffer.

        // iterate over pixels and extract R, G, and B values. Add to bytebuffer.
        var pixel = 0
        for (i in 0 until imageSize) {
            for (j in 0 until imageSize) {
                val `val` = intValues[pixel++] // RGB
                byteBuffer.putFloat((`val` shr 16 and 0xFF) * (1f / 255f))
                byteBuffer.putFloat((`val` shr 8 and 0xFF) * (1f / 255f))
                byteBuffer.putFloat((`val` and 0xFF) * (1f / 255f))
            }
        }

        inputFeature0.loadBuffer(byteBuffer)

        // Runs model inference and gets result.
        val outputs = flowerModel.process(inputFeature0)
        val outputFeature0: TensorBuffer = outputs.outputFeature0AsTensorBuffer
        //inputFeature0.loadBuffer(byteBuffer)
        val confidences = outputFeature0.floatArray

        // find the index of the class with the biggest confidence.
        var maxPos = 0
        var maxConfidence = 0f
        for (i in confidences.indices) {
            if (confidences[i] > maxConfidence) {
                maxConfidence = confidences[i]
                maxPos = i
            }
        }

        var s = ""
        for (i in classes.indices) {
            if(confidences[i] >0.4) {
                items.add(Recognition(classes[i], confidences[i]))
            }

            s += "\n${classes[i]}: ${"%.1f%%".format(confidences[i] * 100f)}%"
            //Log.v("Log", s)
        }

        // Return the result
        listener(items.toList())

        // Close the image,this tells CameraX to feed the next image to the analyzer
        imageProxy.close()
    }



    private fun getImage(image: Bitmap): Bitmap {
        var lImage: Bitmap
        val dimension = image.width.coerceAtMost(image.height)
        lImage = ThumbnailUtils.extractThumbnail(image, dimension, dimension)
        lImage = Bitmap.createScaledBitmap(image, 224, 224, false)
        return lImage
    }


    /**
     * Convert Image Proxy to Bitmap
     */
    private val yuvToRgbConverter = YuvToRgbConverter(ctx)
    private lateinit var bitmapBuffer: Bitmap
    private lateinit var rotationMatrix: Matrix

    @SuppressLint("UnsafeExperimentalUsageError")
    @androidx.annotation.OptIn(androidx.camera.core.ExperimentalGetImage::class)
    private fun toBitmap(imageProxy: ImageProxy): Bitmap? {

        val image = imageProxy.image ?: return null

        // Initialise Buffer
        if (!::bitmapBuffer.isInitialized) {
            // The image rotation and RGB image buffer are initialized only once
            //Log.d(TAG, "Initialise toBitmap()")
            rotationMatrix = Matrix()
            rotationMatrix.postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
            bitmapBuffer = Bitmap.createBitmap(
                imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888
            )
        }

        // Pass image to an image analyser
        yuvToRgbConverter.yuvToRgb(image, bitmapBuffer)

        // Create the Bitmap in the correct orientation
        return Bitmap.createBitmap(
            bitmapBuffer,
            0,
            0,
            bitmapBuffer.width,
            bitmapBuffer.height,
            rotationMatrix,
            false
        )
    }


}
