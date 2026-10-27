package com.datacache.app;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.SurfaceTexture;
import android.os.Bundle;
import android.util.Base64;
import android.view.Gravity;
import android.view.TextureView;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * 얼굴 등록 (동의 인원 한 명): 정면 · 왼쪽 · 오른쪽 사진 3장 → 서버가 얼굴 특징값만 만들고 사진은 바로 지운다.
 * 앞 카메라가 기본, 오른쪽 버튼으로 뒤 카메라 (가족 얼굴을 찍어 줄 때) — 바꿔도 찍은 장은 남는다.
 * 등록된 '본인' 얼굴로 녹화 1단계에서 본인을 확인하고, 녹화 속 참여자를 알아본다. 동의한 가족은 가리지 않고 가상 얼굴로.
 */
public class FaceActivity extends Activity implements TextureView.SurfaceTextureListener {
    private static final String[] PROMPTS = {"정면을 봐 주세요", "고개를 왼쪽으로 살짝 돌려 주세요", "고개를 오른쪽으로 살짝 돌려 주세요"};
    private final List<byte[]> shots = new ArrayList<>();
    private String pid, name;
    private TextureView preview;
    private CameraRecorder camera;
    private TextView step, prompt, msg;
    private boolean front = true, busy;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        pid = getIntent().getStringExtra("pid");
        name = getIntent().getStringExtra("name");
        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(Color.BLACK);
        preview = new TextureView(this);
        preview.setSurfaceTextureListener(this);
        Ui.fitPreview(this, preview);
        root.addView(preview, new FrameLayout.LayoutParams(-1, -1));
        root.addView(new RecordActivity.FaceOval(this), new FrameLayout.LayoutParams(-1, -1));

        LinearLayout top = new LinearLayout(this);
        top.setOrientation(LinearLayout.VERTICAL);
        top.setBackground(Ui.round(0x99000000, Ui.dp(this, 18)));
        top.setPadding(Ui.dp(this, 16), Ui.dp(this, 14), Ui.dp(this, 16), Ui.dp(this, 14));
        top.addView(Ui.text(this, (name == null ? "" : name) + " 님 얼굴 등록", 17, Color.WHITE, true));
        step = Ui.text(this, "", 13, Ui.ACCENT, true);
        step.setPadding(0, Ui.dp(this, 8), 0, 0);
        top.addView(step);
        TextView note = Ui.text(this, "사진은 얼굴 특징값을 만든 뒤 바로 지워져요", 12, 0xCCFFFFFF, false);
        note.setPadding(0, Ui.dp(this, 6), 0, 0);
        top.addView(note);
        FrameLayout.LayoutParams tp = new FrameLayout.LayoutParams(-1, -2, Gravity.TOP);
        tp.setMargins(Ui.dp(this, 12), Ui.dp(this, 40), Ui.dp(this, 12), 0);
        root.addView(top, tp);

        LinearLayout words = new LinearLayout(this);
        words.setOrientation(LinearLayout.VERTICAL);
        prompt = Ui.text(this, "", 20, Color.WHITE, true);
        prompt.setGravity(Gravity.CENTER);
        prompt.setShadowLayer(10, 0, 0, Color.BLACK);
        words.addView(prompt);
        msg = Ui.text(this, "", 14, Color.WHITE, false);
        msg.setGravity(Gravity.CENTER);
        msg.setShadowLayer(10, 0, 0, Color.BLACK);
        msg.setPadding(0, Ui.dp(this, 6), 0, 0);
        words.addView(msg);
        FrameLayout.LayoutParams wp = new FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM);
        wp.bottomMargin = Ui.dp(this, 150);
        root.addView(words, wp);

        FrameLayout bar = new FrameLayout(this);
        bar.setBackgroundColor(0xE6000000);
        RecordRing shoot = new RecordRing(this);
        shoot.setCameraStyle();
        shoot.setContentDescription("찍기");
        shoot.setOnClickListener(v -> capture());
        bar.addView(shoot, new FrameLayout.LayoutParams(Ui.dp(this, 74), Ui.dp(this, 74), Gravity.CENTER));
        FrameLayout sw = new FrameLayout(this);
        sw.setBackground(Ui.pressable(0x55FFFFFF, Ui.dp(this, 26)));
        sw.addView(new Icon(this, Icon.SWITCH, Color.WHITE), new FrameLayout.LayoutParams(Ui.dp(this, 26), Ui.dp(this, 26), Gravity.CENTER));
        sw.setContentDescription("앞뒤 카메라 바꾸기");
        sw.setOnClickListener(v -> { front = !front; reopen(); paint(); });
        FrameLayout.LayoutParams sp = new FrameLayout.LayoutParams(Ui.dp(this, 52), Ui.dp(this, 52), Gravity.END | Gravity.CENTER_VERTICAL);
        sp.rightMargin = Ui.dp(this, 36);
        bar.addView(sw, sp);
        root.addView(bar, new FrameLayout.LayoutParams(-1, Ui.dp(this, 130), Gravity.BOTTOM));
        setContentView(root);
        paint();
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, 1);
        }
    }

    private void paint() {
        int k = Math.min(shots.size(), PROMPTS.length - 1);
        step.setText(Math.min(shots.size() + 1, 3) + " / 3");
        prompt.setText(PROMPTS[k]);
        msg.setText("얼굴이 동그라미 안에 오게" + (front ? "" : "\n(뒤 카메라 · 가족 얼굴을 찍어 주세요)"));
    }

    @Override
    public void onRequestPermissionsResult(int code, String[] perms, int[] res) {
        if (preview.isAvailable()) open(preview.getSurfaceTexture());
    }

    @Override public void onSurfaceTextureAvailable(SurfaceTexture st, int w, int h) { open(st); }
    @Override public void onSurfaceTextureSizeChanged(SurfaceTexture st, int w, int h) { Ui.applyFit(this, preview); }
    @Override public boolean onSurfaceTextureDestroyed(SurfaceTexture st) { return true; }
    @Override public void onSurfaceTextureUpdated(SurfaceTexture st) {}

    private void open(SurfaceTexture st) {
        if (camera != null || checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return;
        camera = new CameraRecorder(this);
        camera.open(st, new CameraRecorder.Ready() {
            @Override public void ok() { runOnUiThread(() -> Ui.applyFit(FaceActivity.this, preview)); }
            @Override public void fail(String why) { runOnUiThread(() -> msg.setText("카메라: " + why)); }
        }, front);
    }

    private void reopen() {
        if (camera != null) camera.close();
        camera = null;
        if (preview.isAvailable()) open(preview.getSurfaceTexture());
    }

    private void capture() {
        if (busy || camera == null) return;
        if (shots.size() >= PROMPTS.length) { upload(); return; }     // 올리기 실패 뒤 다시 누르면 다시 올린다
        int w = 720, h = Math.round(720f * Math.max(1, preview.getHeight()) / Math.max(1, preview.getWidth()));
        Bitmap b = preview.getBitmap(w, h);
        if (b == null) return;
        ByteArrayOutputStream jpg = new ByteArrayOutputStream();
        b.compress(Bitmap.CompressFormat.JPEG, 92, jpg);
        b.recycle();
        shots.add(jpg.toByteArray());
        if (shots.size() < PROMPTS.length) { paint(); return; }
        upload();
    }

    private void upload() {
        busy = true;
        prompt.setText("올리는 중…");
        msg.setText("");
        new Thread(() -> {
            try {
                JSONArray arr = new JSONArray();
                for (byte[] s : shots) arr.put(Base64.encodeToString(s, Base64.NO_WRAP));
                JSONObject r = Uploader.callJson(this, "POST", "/api/people/" + pid + "/face", new JSONObject().put("photos_b64", arr));
                shots.clear();       // 폰에도 남기지 않는다
                runOnUiThread(() -> {
                    busy = false;
                    if ("enrolled".equals(r.optString("face"))) {
                        Toast.makeText(this, "얼굴을 등록했어요", Toast.LENGTH_SHORT).show();
                        setResult(RESULT_OK);
                        finish();
                    } else {
                        paint();
                        prompt.setText(r.optString("face_error", "다시 찍어 주세요"));
                        msg.setText("정면부터 다시 세 장을 찍어 주세요");
                    }
                });
            } catch (Exception e) {
                runOnUiThread(() -> {
                    busy = false;
                    prompt.setText("올리지 못했어요 · " + Uploader.reason(e));
                    msg.setText("찍기 버튼을 누르면 다시 올려요");
                });
            }
        }).start();
    }

    @Override
    protected void onDestroy() {
        if (camera != null) camera.close();
        super.onDestroy();
    }
}
