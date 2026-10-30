package com.datacash.app;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.widget.GridLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import org.json.JSONObject;

import java.io.File;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Locale;

/**
 * 가상 얼굴 미리보기: 판매용 영상(동의 안 한 사람·화면은 가리고, 나와 동의한 사람 얼굴은 가상 인물로 바꾼 것)에서
 * 고르게 뽑은 6장면 + 치환 결과 요약 (몇 장면을 바꿨는지, 재식별 검사). 원본 영상은 보여 주지 않는다.
 */
public class PreviewActivity extends BaseActivity {
    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        File dir = new File(getIntent().getStringExtra("dir"));
        LinearLayout body = Ui.page(this, getString(R.string.preview_title), true);
        JSONObject st = Uploader.localState(dir);
        LinearLayout c = Ui.card(this);
        c.addView(Ui.text(this, st.optString("title", getString(R.string.recording)), 17, Ui.INK, true));
        TextView sum = Ui.text(this, getString(R.string.preview_sum), 13, Ui.SUB, false);
        sum.setPadding(0, Ui.dp(this, 6), 0, 0);
        c.addView(sum);
        JSONObject fs = st.optJSONObject("privacy") == null ? null : st.optJSONObject("privacy").optJSONObject("face_swap");
        if (fs != null) {
            TextView t = Ui.text(this, getString(R.string.preview_stats,
                    fs.optInt("frames"), fs.optInt("swapped"), fs.optInt("small_face_swapped"), fs.optInt("reid_blocked"), fs.optDouble("reid_max")),
                    13, Ui.INK, true);
            t.setPadding(0, Ui.dp(this, 10), 0, 0);
            c.addView(t);
        }
        Ui.add(this, body, c);
        GridLayout g = new GridLayout(this);
        g.setColumnCount(2);
        int cell = (getResources().getDisplayMetrics().widthPixels - Ui.dp(this, 32 + 12)) / 2;
        for (int k = 0; k < 6; k++) {
            ImageView iv = new ImageView(this);
            iv.setScaleType(ImageView.ScaleType.CENTER_CROP);
            iv.setBackground(Ui.round(Ui.LINE, Ui.dp(this, 14)));
            iv.setClipToOutline(true);
            GridLayout.LayoutParams lp = new GridLayout.LayoutParams();
            lp.width = cell;
            lp.height = cell * 3 / 4;
            lp.setMargins(0, 0, k % 2 == 0 ? Ui.dp(this, 12) : 0, Ui.dp(this, 12));
            g.addView(iv, lp);
            int kk = k;
            new Thread(() -> {
                try {
                    HttpURLConnection h = (HttpURLConnection) new URL(Store.server(this) + "/api/sessions/" + dir.getName() + "/preview/" + kk).openConnection();
                    h.setRequestProperty("X-DC-Token", Store.token(this));
                    h.setRequestProperty("X-DC-Device", Store.deviceId(this));
                    if (h.getResponseCode() != 200) {
                        if (kk == 0) runOnUiThread(() -> sum.setText(R.string.preview_pending));
                        return;
                    }
                    try (InputStream in = h.getInputStream()) {
                        Bitmap b = BitmapFactory.decodeStream(in);
                        runOnUiThread(() -> iv.setImageBitmap(b));
                    }
                } catch (Exception ignored) {
                }
            }).start();
        }
        body.addView(g);
    }
}
