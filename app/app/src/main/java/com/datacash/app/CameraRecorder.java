package com.datacash.app;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Rect;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Range;
import android.view.Surface;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Camera2 → MediaRecorder(H.264). 장면마다 촬영 시각(SENSOR_TIMESTAMP)을 video_frames.csv 에 적는다.
 * 시각 기준이 REALTIME 인 카메라는 BOOTTIME(폰의 다른 기록과 같은 시계), 아니면 meta 에 표시한다.
 * 앞/뒤 카메라, 배율(두 손가락·배율 버튼 — 녹화 중 바뀌면 시각을 남긴다), 화면 방향에 맞춘 영상 회전.
 */
final class CameraRecorder {
    private final Context ctx;
    private HandlerThread thread;
    private Handler handler;
    private CameraDevice camera;
    private CameraCaptureSession session;
    private CaptureRequest.Builder request;
    private MediaRecorder recorder;
    private Surface previewSurface;
    private final List<Long> frameNs = new ArrayList<>();
    private final JSONArray zoomLog = new JSONArray();
    private String timestampSource = "unknown";
    private final int width = 1280, height = 720, fps = 30;
    private boolean recording, front;
    private int sensorOrientation = 90, orientationHint = 90;
    private float zoom = 1f, maxZoom = 1f, minZoom = 1f;
    private Rect active;

    CameraRecorder(Context c) {
        ctx = c;
    }

    interface Ready { void ok(); void fail(String why); }

    void open(SurfaceTexture preview, Ready ready) {
        open(preview, ready, false);
    }

    boolean isFront() { return front; }
    float zoom() { return zoom; }
    float minZoom() { return minZoom; }
    float maxZoom() { return maxZoom; }

    /** front 면 앞 카메라. */
    @SuppressLint("MissingPermission")
    void open(SurfaceTexture preview, Ready ready, boolean front) {
        this.front = front;
        thread = new HandlerThread("camera");
        thread.start();
        handler = new Handler(thread.getLooper());
        try {
            CameraManager cm = (CameraManager) ctx.getSystemService(Context.CAMERA_SERVICE);
            String id = null;
            int want = front ? CameraCharacteristics.LENS_FACING_FRONT : CameraCharacteristics.LENS_FACING_BACK;
            for (String c : cm.getCameraIdList()) {
                Integer f = cm.getCameraCharacteristics(c).get(CameraCharacteristics.LENS_FACING);
                if (f != null && f == want) { id = c; break; }
            }
            if (id == null) id = cm.getCameraIdList()[0];
            CameraCharacteristics ch = cm.getCameraCharacteristics(id);
            Integer src = ch.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE);
            timestampSource = src != null && src == CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME ? "BOOTTIME" : "unknown";
            Integer so = ch.get(CameraCharacteristics.SENSOR_ORIENTATION);
            sensorOrientation = so == null ? 90 : so;
            active = ch.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE);
            zoom = 1f;
            minZoom = 1f;
            Float mz = ch.get(CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM);
            maxZoom = mz == null ? 1f : Math.min(mz, 10f);
            if (Build.VERSION.SDK_INT >= 30) {
                Range<Float> zr = ch.get(CameraCharacteristics.CONTROL_ZOOM_RATIO_RANGE);
                if (zr != null) { minZoom = zr.getLower(); maxZoom = Math.min(zr.getUpper(), 10f); }
            }
            preview.setDefaultBufferSize(width, height);
            previewSurface = new Surface(preview);
            cm.openCamera(id, new CameraDevice.StateCallback() {
                @Override public void onOpened(CameraDevice d) { camera = d; startPreview(ready); }
                @Override public void onDisconnected(CameraDevice d) { d.close(); }
                @Override public void onError(CameraDevice d, int e) { d.close(); ready.fail(ctx.getString(R.string.camera_error, e)); }
            }, handler);
        } catch (Exception e) {
            ready.fail(e.getMessage());
        }
    }

    private void applyZoom(CaptureRequest.Builder b) {
        if (Build.VERSION.SDK_INT >= 30) {
            b.set(CaptureRequest.CONTROL_ZOOM_RATIO, zoom);
        } else if (active != null) {
            float z = Math.max(1f, zoom);
            int w = Math.round(active.width() / z), h = Math.round(active.height() / z);
            int x = active.centerX() - w / 2, y = active.centerY() - h / 2;
            b.set(CaptureRequest.SCALER_CROP_REGION, new Rect(x, y, x + w, y + h));
        }
    }

    /** 배율 바꾸기 (범위 안으로). 녹화 중이면 시각과 배율을 남긴다. */
    void setZoom(float z) {
        float nz = Math.max(minZoom, Math.min(maxZoom, z));
        if (Math.abs(nz - zoom) < 0.01f) return;
        zoom = nz;
        if (recording) {
            try { zoomLog.put(new JSONObject().put("t_ns", Clock.now()).put("zoom", Math.round(zoom * 100) / 100.0)); } catch (Exception ignored) {}
        }
        handler.post(() -> {
            try {
                if (request == null || session == null) return;
                applyZoom(request);
                session.setRepeatingRequest(request.build(), recording ? frameCallback : null, handler);
            } catch (Exception ignored) {
            }
        });
    }

    private final CameraCaptureSession.CaptureCallback frameCallback = new CameraCaptureSession.CaptureCallback() {
        @Override public void onCaptureStarted(CameraCaptureSession ss, CaptureRequest r, long ts, long frame) {
            if (recording) synchronized (frameNs) { frameNs.add(ts); }
        }
    };

    private void startPreview(Ready ready) {
        try {
            camera.createCaptureSession(Arrays.asList(previewSurface), new CameraCaptureSession.StateCallback() {
                @Override public void onConfigured(CameraCaptureSession s) {
                    session = s;
                    try {
                        request = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
                        request.addTarget(previewSurface);
                        applyZoom(request);
                        s.setRepeatingRequest(request.build(), null, handler);
                        ready.ok();
                    } catch (Exception e) { ready.fail(e.getMessage()); }
                }
                @Override public void onConfigureFailed(CameraCaptureSession s) { ready.fail(ctx.getString(R.string.camera_preview_fail)); }
            }, handler);
        } catch (Exception e) {
            ready.fail(e.getMessage());
        }
    }

    /** displayRotation: Surface.ROTATION_* (녹화 시작 때 화면 방향) → 영상 회전 힌트. */
    void start(File dir, int displayRotation, Ready ready) {
        int deg = new int[]{0, 90, 180, 270}[displayRotation & 3];
        orientationHint = front ? (sensorOrientation + deg) % 360 : (sensorOrientation - deg + 360) % 360;
        try {
            recorder = new MediaRecorder(ctx);
            recorder.setVideoSource(MediaRecorder.VideoSource.SURFACE);
            recorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
            recorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264);
            recorder.setVideoEncodingBitRate(8_000_000);
            recorder.setVideoFrameRate(fps);
            recorder.setVideoSize(width, height);
            recorder.setOrientationHint(orientationHint);
            recorder.setOutputFile(new File(dir, "video.mp4").getAbsolutePath());
            recorder.prepare();
            Surface rec = recorder.getSurface();
            frameNs.clear();
            session.close();
            camera.createCaptureSession(Arrays.asList(previewSurface, rec), new CameraCaptureSession.StateCallback() {
                @Override public void onConfigured(CameraCaptureSession s) {
                    session = s;
                    try {
                        request = camera.createCaptureRequest(CameraDevice.TEMPLATE_RECORD);
                        request.addTarget(previewSurface);
                        request.addTarget(rec);
                        request.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, new Range<>(fps, fps));
                        applyZoom(request);
                        recording = true;
                        try { zoomLog.put(new JSONObject().put("t_ns", Clock.now()).put("zoom", Math.round(zoom * 100) / 100.0)); } catch (Exception ignored) {}
                        s.setRepeatingRequest(request.build(), frameCallback, handler);
                        recorder.start();
                        ready.ok();
                    } catch (Exception e) { recording = false; ready.fail(e.getMessage()); }
                }
                @Override public void onConfigureFailed(CameraCaptureSession s) { ready.fail(ctx.getString(R.string.camera_session_fail)); }
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
                .put("frames_logged", frameNs.size()).put("timestamp_source", timestampSource)
                .put("orientation_hint", orientationHint).put("camera", front ? "front" : "back").put("zoom_log", zoomLog);
    }

    void close() {
        try { if (session != null) session.close(); } catch (Exception ignored) {}
        try { if (camera != null) camera.close(); } catch (Exception ignored) {}
        if (recorder != null) { try { recorder.release(); } catch (Exception ignored) {} }
        if (thread != null) thread.quitSafely();
        session = null;
        camera = null;
        request = null;
    }
}
