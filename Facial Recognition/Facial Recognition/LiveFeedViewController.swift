//
//  LiveViewController.swift
//  Facial Recognition
//
//  Created by Donald Dang on 2022-01-02.
//

import AVFoundation
import UIKit
import Vision

class LiveFeedViewController: UIViewController {
    
    private let captureSession = AVCaptureSession()
    private lazy var previewLayer = AVCaptureVideoPreviewLayer(session: self.captureSession)
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private var faceLayers: [CAShapeLayer] = []

    override func viewDidLoad() {
        super.viewDidLoad()
        setupCamera()
        captureSession.startRunning()
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        self.previewLayer.frame = self.view.frame
    }
    
    private func setupCamera() { //This uses AVFoundation to find our camera
        let deviceDiscoverySession = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: .video, position: .front)
        if let device = deviceDiscoverySession.devices.first { //we want to launch the front camera, if it exists
            if let deviceInput = try? AVCaptureDeviceInput(device: device) {
                if captureSession.canAddInput(deviceInput) {
                    captureSession.addInput(deviceInput)
                    
                    setupPreview()
                }
            }
        }
    }
    
    private func setupPreview() { //launch our screen to eventually capture output
        self.previewLayer.videoGravity = .resizeAspectFill
        self.view.layer.addSublayer(self.previewLayer)
        self.previewLayer.frame = self.view.frame
        
        self.videoDataOutput.videoSettings = [(kCVPixelBufferPixelFormatTypeKey as NSString) : NSNumber(value: kCVPixelFormatType_32BGRA)] as [String : Any] //set the data output values

        self.videoDataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "camera queue"))
        self.captureSession.addOutput(self.videoDataOutput)
        
        let videoConnection = self.videoDataOutput.connection(with: .video)
        videoConnection?.videoOrientation = .portrait //set orientation of our video
    }
}

extension LiveFeedViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) { //we want to create the rectangle for the facial recognition, followed by the eventual details we need to capture (ie. left eye, right eye, etc.)
        
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
          return
        }

        let faceDetectionRequest = VNDetectFaceLandmarksRequest(completionHandler: { (request: VNRequest, error: Error?) in
            DispatchQueue.main.async {
                self.faceLayers.forEach({ drawing in drawing.removeFromSuperlayer() }) //remove layers

                if let observations = request.results as? [VNFaceObservation] {
                    self.handleFaceDetectionObservations(observations: observations)
                }
            }
        })

        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: imageBuffer, orientation: .leftMirrored, options: [:]) //remember, front cameras are always reversed!

        do {
            try imageRequestHandler.perform([faceDetectionRequest])
        } catch {
          print(error.localizedDescription)
        }
    }
    
    private func handleFaceDetectionObservations(observations: [VNFaceObservation]) { //used for the 'box' for our facial recognition module
        for observation in observations {
            let faceRectConverted = self.previewLayer.layerRectConverted(fromMetadataOutputRect: observation.boundingBox)
            let faceRectanglePath = CGPath(rect: faceRectConverted, transform: nil)
            
            let faceLayer = CAShapeLayer()
            faceLayer.path = faceRectanglePath
            faceLayer.fillColor = UIColor.clear.cgColor
            faceLayer.strokeColor = UIColor.orange.cgColor
            
            self.faceLayers.append(faceLayer)
            self.view.layer.addSublayer(faceLayer)
            
            if let landmarks = observation.landmarks { //in this section we acquire all of the details of the facial recognition module
                if let leftEye = landmarks.leftEye {
                    self.handleLandmark(leftEye, faceBoundingBox: faceRectConverted)
                }
                if let leftEyebrow = landmarks.leftEyebrow {
                    self.handleLandmark(leftEyebrow, faceBoundingBox: faceRectConverted)
                }
                if let rightEye = landmarks.rightEye {
                    self.handleLandmark(rightEye, faceBoundingBox: faceRectConverted)
                }
                if let rightEyebrow = landmarks.rightEyebrow {
                    self.handleLandmark(rightEyebrow, faceBoundingBox: faceRectConverted)
                }

                if let nose = landmarks.nose {
                    self.handleLandmark(nose, faceBoundingBox: faceRectConverted)
                }

                if let outerLips = landmarks.outerLips {
                    self.handleLandmark(outerLips, faceBoundingBox: faceRectConverted)
                }
                if let innerLips = landmarks.innerLips {
                    self.handleLandmark(innerLips, faceBoundingBox: faceRectConverted)
                }
            }
        }
    }
    
    private func handleLandmark(_ eye: VNFaceLandmarkRegion2D, faceBoundingBox: CGRect) { //we add a 'layer' on top of our video layer (the layer used to record video) and add our eye, nose, mouth, etc.
        let landmarkPath = CGMutablePath()
        let landmarkPathPoints = eye.normalizedPoints
            .map({ eyePoint in
                CGPoint(
                    x: eyePoint.y * faceBoundingBox.height + faceBoundingBox.origin.x, //This is a concept used in image processing. We multiply the y coordinate with the height of the eye, for example, and add the origin of x. This is the same vice versa with the second line
                    y: eyePoint.x * faceBoundingBox.width + faceBoundingBox.origin.y)
            })
        landmarkPath.addLines(between: landmarkPathPoints)
        landmarkPath.closeSubpath()
        let landmarkLayer = CAShapeLayer()
        landmarkLayer.path = landmarkPath
        landmarkLayer.fillColor = UIColor.clear.cgColor
        landmarkLayer.strokeColor = UIColor.cyan.cgColor

        self.faceLayers.append(landmarkLayer)
        self.view.layer.addSublayer(landmarkLayer)
    }
}
