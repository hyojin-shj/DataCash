package com.datacache.app;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.os.Bundle;
import android.view.Gravity;
import android.view.TextureView;
import android.view.View;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;

/**
 * 얼굴 찍기 — 세 가지 쓰임.
 *   owner   내 얼굴 등록 (처음 한 번): 서버가 특징값만 저장하고 사진은 버린다. 녹화 속 참여자를 이 얼굴로 고른다.
 *   check   녹화 전 1단계 얼굴 확인: 등록한 본인인지 (동의한 본인이 찍는지)
 *   person  동의 인원 사진: 찍은 JPEG 를 돌려준다 (동의 인원 화면이 이름·동의와 함께 올린다)
 * 타원 안내선 안에 얼굴을 두고 버튼을 누르면 미리보기에서 한 장을 떠서 보낸다.
 */
public class FaceActivity extends Activity implements TextureView.SurfaceTextureListener {
    static final String OWNER = "owner", CHECK = "check", PERSON = "person";
    static byte[] lastPhoto;   // person 모드 결과 (인텐트로 넘기기엔 커서)

    private String mode;
    private TextureView preview;
    private CameraRecorder camera;
    private TextView msg;
    private RecordRing shoot;
    private boolean front = true, busy;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        mode = getIntent().getStringExtra("mode");
        if (mode == null) mode = OWNER;
        front = !PERSON.equals(mode);           // 가족은 폰 주인이 뒤 카메라로 찍어 주는 경우가 많다
        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(Color.BLACK);
        preview = new TextureView(this);
        preview.setSurfaceTextureListener(this);
        root.addView(preview, new FrameLayout.LayoutParams(-1, -1));
        root.addView(new Oval(this), new FrameLayout.LayoutParams(-1, -1));

        LinearLayout top = new LinearLayout(this);
        top.setOrientation(LinearLayout.VERTICAL);
        top.setBackground(Ui.round(0x99000000, Ui.dp(this, 18)));
        top.setPadding(Ui.dp(this, 16), Ui.dp(this, 14), Ui.dp(this, 16), Ui.dp(this, 14));
        String title = OWNER.equals(mode) ? "내 얼굴 등록 (처음 한 번)" : CHECK.equals(mode) ? "1단계 얼굴 확인" : "함께 찍힐 사람 얼굴";
        top.addView(Ui.text(this, title, 17, Color.WHITE, true));
        String why = OWNER.equals(mode) ? "녹화 속에서 나를 알아보는 데만 써요. 사진은 저장하지 않고 특징값만 남아요."
                : CHECK.equals(mode) ? "1단계 얼굴 확인 → 2단계 동작 촬영 · 동의한 본인이 찍는지 확인해요"
                : "동의한 사람은 가리지 않고 가상 얼굴로만 바꿔요. 사진은 저장하지 않아요.";
        TextView w = Ui.text(this, why, 13, Ui.ACCENT, true);
        w.setPadding(0, Ui.dp(this, 8), 0, 0);
        top.addView(w);
        FrameLayout.LayoutParams tp = new FrameLayout.LayoutParams(-1, -2, Gravity.TOP);
        tp.setMargins(Ui.dp(this, 12), Ui.dp(this, 40), Ui.dp(this, 12), 0);
        root.addView(top, tp);

        FrameLayout bar = new FrameLayout(this);
        bar.setBackgroundColor(0xE6000000);
        msg = Ui.text(this, "타원 안에 얼굴을 맞춰 주세요", 13, Color.WHITE, false);
        msg.setGravity(Gravity.CENTER);
        FrameLayout.LayoutParams mp = new FrameLayout.LayoutParams(-1, -2, Gravity.TOP);
        mp.topMargin = Ui.dp(this, 12);
        bar.addView(msg, mp);
        shoot = new RecordRing(this);
        shoot.setCameraStyle();
        shoot.setContentDescription("찍기");
        shoot.setOnClickListener(v -> capture());
        FrameLayout.LayoutParams sp = new FrameLayout.LayoutParams(Ui.dp(this, 72), Ui.dp(this, 72), Gravity.CENTER_HORIZONTAL | Gravity.BOTTOM);
        sp.bottomMargin = Ui.dp(this, 22);
        bar.addView(shoot, sp);
        TextView flip = Ui.text(this, "앞뒤", 13, Color.WHITE, true);
        flip.setGravity(Gravity.CENTER);
        flip.setBackground(Ui.pressable(0x33FFFFFF, Ui.dp(this, 22)));
        flip.setPadding(Ui.dp(this, 14), Ui.dp(this, 10), Ui.dp(this, 14), Ui.dp(this, 10));
        flip.setOnClickListener(v -> { front = !front; reopen(); });
        FrameLayout.LayoutParams fp = new FrameLayout.LayoutParams(-2, -2, Gravity.END | Gravity.BOTTOM);
        fp.setMargins(0, 0, Ui.dp(this, 24), Ui.dp(this, 36));
        bar.addView(flip, fp);
        if (CHECK.equals(mode)) {   // 서버에 닿지 않을 때만 쓰는 길 (참여자는 크기·손으로 고른다)
            TextView skip = Ui.text(this, "나중에", 13, 0xAAFFFFFF, true);
            skip.setPadding(Ui.dp(this, 14), Ui.dp(this, 10), Ui.dp(this, 14), Ui.dp(this, 10));
            skip.setOnClickListener(v -> done(RESULT_FIRST_USER, null));
            FrameLayout.LayoutParams kp = new FrameLayout.LayoutParams(-2, -2, Gravity.START | Gravity.BOTTOM);
            kp.setMargins(Ui.dp(this, 24), 0, 0, Ui.dp(this, 36));
            bar.addView(skip, kp);
        }
        root.addView(bar, new FrameLayout.LayoutParams(-1, Ui.dp(this, 150), Gravity.BOTTOM));
        setContentView(root);
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, 1);
        }
    }

    @Override
    public void onRequestPermissionsResult(int code, String[] perms, int[] res) {
        if (preview.isAvailable()) open(preview.getSurfaceTexture());
    }

    @Override public void onSurfaceTextureAvailable(SurfaceTexture st, int w, int h) { open(st); }
    @Override public void onSurfaceTextureSizeChanged(SurfaceTexture st, int w, int h) {}
    @Override public boolean onSurfaceTextureDestroyed(SurfaceTexture st) { return true; }
    @Override public void onSurfaceTextureUpdated(SurfaceTexture st) {}

    private void open(SurfaceTexture st) {
        if (camera != null || checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return;
        camera = new CameraRecorder(this);
        camera.open(st, new CameraRecorder.Ready() {
            @Override public void ok() {}
            @Override public void fail(String why) { runOnUiThread(() -> msg.setText("카메라: " + why)); }
        }, front);
        Ui.fitPreview(preview, front);           // 비율 유지, 앞 카메라는 거울처럼
    }

    private void reopen() {
        if (camera != null) camera.close();
        camera = null;
        if (preview.isAvailable()) open(preview.getSurfaceTexture());
    }

    private void capture() {
        if (busy || camera == null) return;
        Bitmap b = preview.getBitmap(720, 720 * Math.max(1, preview.getHeight()) / Math.max(1, preview.getWidth()));
        if (b == null) return;
        ByteArrayOutputStream jpg = new ByteArrayOutputStream();
        b.compress(Bitmap.CompressFormat.JPEG, 90, jpg);
        byte[] data = jpg.toByteArray();
        if (PERSON.equals(mode)) {
            lastPhoto = data;
            done(RESULT_OK, null);
            return;
        }
        busy = true;
        msg.setText("확인하는 중…");
        new Thread(() -> {
            try {
                JSONObject r = Uploader.call(this, "POST", OWNER.equals(mode) ? "/api/face/owner" : "/api/face/check", data, "image/jpeg");
                runOnUiThread(() -> {
                    busy = false;
                    if (OWNER.equals(mode)) {
                        Store.setOwnerFace(this, true);
                        done(RESULT_OK, r);
                    } else if (!r.optBoolean("registered")) {
                        Store.setOwnerFace(this, false);     // 서버에 등록이 없다 → 등록부터
                        mode = OWNER;
                        msg.setText("아직 얼굴이 등록되지 않았어요 — 한 번 더 찍으면 등록돼요");
                    } else if (r.optBoolean("match")) {
                        done(RESULT_OK, r);
                    } else {
                        msg.setText("등록한 본인과 달라 보여요. 밝은 곳에서 정면으로 다시 찍어 주세요");
                    }
                });
            } catch (Exception e) {
                runOnUiThread(() -> { busy = false; msg.setText(Uploader.reason(e)); });
            }
        }).start();
    }

    private void done(int code, JSONObject r) {
        Intent i = new Intent();
        if (r != null) i.putExtra("similarity", r.optDouble("similarity", 0));
        setResult(code, i);
        finish();
    }

    @Override
    protected void onDestroy() {
        if (camera != null) camera.close();
        super.onDestroy();
    }

    /** 얼굴 자리 타원 안내선 (바깥은 살짝 어둡게). */
    private static final class Oval extends View {
        private final Paint dim = new Paint(), line = new Paint(Paint.ANTI_ALIAS_FLAG), clear = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final RectF r = new RectF();

        Oval(Context c) {
            super(c);
            setLayerType(LAYER_TYPE_SOFTWARE, null);
            dim.setColor(0x66000000);
            clear.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.CLEAR));
            line.setStyle(Paint.Style.STROKE);
            line.setColor(Color.WHITE);
            line.setStrokeWidth(3 * c.getResources().getDisplayMetrics().density);
        }

        @Override
        protected void onDraw(Canvas c) {
            float w = getWidth(), h = getHeight(), ow = w * 0.62f, oh = ow * 1.3f;
            r.set((w - ow) / 2, h * 0.42f - oh / 2, (w + ow) / 2, h * 0.42f + oh / 2);
            c.drawRect(0, 0, w, h, dim);
            c.drawOval(r, clear);
            c.drawOval(r, line);
        }
    }
}
