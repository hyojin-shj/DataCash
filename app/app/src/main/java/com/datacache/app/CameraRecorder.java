package com.datacache.app;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.TotalCaptureResult;
import android.media.MediaRecorder;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Range;
import android.view.Surface;

import org.json.JSONObject;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Camera2 → MediaRecorder(H.264). 장면마다 촬영 시각(SENSOR_TIMESTAMP)을 video_frames.csv 에 적는다.
 * 시각 기준이 REALTIME 인 카메라는 BOOTTIME(폰의 다른 기록과 같은 시계), 아니면 meta 에 표시한다.
 */
final class CameraRecorder {
    private final Context ctx;
    private HandlerThread thread;
    private Handler handler;
    private CameraDevice camera;
    private CameraCaptureSession session;
    private MediaRecorder recorder;
    private Surface previewSurface;
    private final List<Long> frameNs = new ArrayList<>();
    private String timestampSource = "unknown";
    private int width = 1280, height = 720, fps = 30;
    private boolean recording;

    CameraRecorder(Context c) {
        ctx = c;
    }

    interface Ready { void ok(); void fail(String why); }

    @SuppressLint("MissingPermission")
    void open(SurfaceTexture preview, Ready ready) {
        thread = new HandlerThread("camera");
        thread.start();
        handler = new Handler(thread.getLooper());
        try {
            CameraManager cm = (CameraManager) ctx.getSystemService(Context.CAMERA_SERVICE);
            String id = null;
            for (String c : cm.getCameraIdList()) {
                Integer f = cm.getCameraCharacteristics(c).get(CameraCharacteristics.LENS_FACING);
                if (f != null && f == CameraCharacteristics.LENS_FACING_BACK) { id = c; break; }
            }
            if (id == null) id = cm.getCameraIdList()[0];
            CameraCharacteristics ch = cm.getCameraCharacteristics(id);
            Integer src = ch.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE);
            timestampSource = src != null && src == CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME ? "BOOTTIME" : "unknown";
            preview.setDefaultBufferSize(width, height);
            previewSurface = new Surface(preview);
            cm.openCamera(id, new CameraDevice.StateCallback() {
                @Override public void onOpened(CameraDevice d) { camera = d; startPreview(ready); }
                @Override public void onDisconnected(CameraDevice d) { d.close(); }
                @Override public void onError(CameraDevice d, int e) { d.close(); ready.fail("카메라 오류 " + e); }
            }, handler);
        } catch (Exception e) {
            ready.fail(e.getMessage());
        }
    }

    private void startPreview(Ready ready) {
        try {
            camera.createCaptureSession(Arrays.asList(previewSurface), new CameraCaptureSession.StateCallback() {
                @Override public void onConfigured(CameraCaptureSession s) {
                    session = s;
                    try {
                        CaptureRequest.Builder b = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
                        b.addTarget(previewSurface);
                        s.setRepeatingRequest(b.build(), null, handler);
                        ready.ok();
                    } catch (Exception e) { ready.fail(e.getMessage()); }
                }
                @Override public void onConfigureFailed(CameraCaptureSession s) { ready.fail("미리보기 실패"); }
            }, handler);
        } catch (Exception e) {
            ready.fail(e.getMessage());
        }
    }

    void start(File dir, Ready ready) {
        try {
            recorder = new MediaRecorder(ctx);
            recorder.setVideoSource(MediaRecorder.VideoSource.SURFACE);
            recorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
            recorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264);
            recorder.setVideoEncodingBitRate(8_000_000);
            recorder.setVideoFrameRate(fps);
            recorder.setVideoSize(width, height);
            recorder.setOrientationHint(90);
            recorder.setOutputFile(new File(dir, "video.mp4").getAbsolutePath());
            recorder.prepare();
            Surface rec = recorder.getSurface();
            frameNs.clear();
            session.close();
            camera.createCaptureSession(Arrays.asList(previewSurface, rec), new CameraCaptureSession.StateCallback() {
                @Override public void onConfigured(CameraCaptureSession s) {
                    session = s;
                    try {
                        CaptureRequest.Builder b = camera.createCaptureRequest(CameraDevice.TEMPLATE_RECORD);
                        b.addTarget(previewSurface);
                        b.addTarget(rec);
                        b.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, new Range<>(fps, fps));
                        s.setRepeatingRequest(b.build(), new CameraCaptureSession.CaptureCallback() {
                            @Override public void onCaptureStarted(CameraCaptureSession ss, CaptureRequest r, long ts, long frame) {
                                if (recording) synchronized (frameNs) { frameNs.add(ts); }
                            }
                        }, handler);
                        recorder.start();
                        recording = true;
                        ready.ok();
                    } catch (Exception e) { ready.fail(e.getMessage()); }
                }
                @Override public void onConfigureFailed(CameraCaptureSession s) { ready.fail("녹화 세션 실패"); }
            }, handler);
        } catch (Exception e) {
            ready.fail(e.getMessage());
        }
    }

    /** 녹화를 멈추고 장면 시각을 적는다. 인코더가 버린 장면이 있으면 개수가 영상과 다를 수 있다 → 서버가 맞춘다. */
    JSONObject stop(File dir) throws Exception {
        recording = false;
        try { session.stopRepeating(); } catch (Exception ignored) {}
        recorder.stop();
        recorder.release();
        recorder = null;
        try (FileWriter w = new FileWriter(new File(dir, "video_frames.csv"))) {
            w.write("frame,t_ns\n");
            synchronized (frameNs) {
                for (int i = 0; i < frameNs.size(); i++) w.write(i + "," + frameNs.get(i) + "\n");
            }
        }
        return new JSONObject().put("width", width).put("height", height).put("fps", fps)
                .put("frames_logged", frameNs.size()).put("timestamp_source", timestampSource).put("orientation_hint", 90);
    }

    void close() {
        try { if (session != null) session.close(); } catch (Exception ignored) {}
        try { if (camera != null) camera.close(); } catch (Exception ignored) {}
        if (recorder != null) { try { recorder.release(); } catch (Exception ignored) {} }
        if (thread != null) thread.quitSafely();
    }
}
