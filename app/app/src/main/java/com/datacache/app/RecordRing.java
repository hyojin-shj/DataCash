package com.datacache.app;

import android.animation.ValueAnimator;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.view.View;

/**
 * 빨간 녹화 버튼. 홈의 오늘 미션 카드에서는 바깥에 진행 고리(찍은 상황 수)와 숨 쉬듯 퍼지는 빛,
 * 촬영 화면에서는 흰 테두리 안의 빨간 원 (녹화 중이면 둥근 네모로 바뀐다).
 */
final class RecordRing extends View {
    private final Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final RectF r = new RectF();
    private float progress = -1f;      // 0~1 이면 바깥 진행 고리를 그린다
    private boolean glow, recording, camera;
    private float pulse;
    private ValueAnimator anim;

    RecordRing(Context c) {
        super(c);
        setClickable(true);
    }

    /** 홈 카드용: 진행 고리 + 빛. */
    void setProgress(float frac) {
        progress = frac;
        glow = true;
        invalidate();
    }

    /** 촬영 화면용: 흰 테두리 버튼. */
    void setCameraStyle() {
        camera = true;
        invalidate();
    }

    void setRecording(boolean on) {
        recording = on;
        invalidate();
    }

    @Override
    protected void onAttachedToWindow() {
        super.onAttachedToWindow();
        anim = ValueAnimator.ofFloat(0f, 1f);
        anim.setDuration(1800);
        anim.setRepeatCount(ValueAnimator.INFINITE);
        anim.addUpdateListener(a -> { pulse = (float) a.getAnimatedValue(); if (glow) invalidate(); });
        anim.start();
    }

    @Override
    protected void onDetachedFromWindow() {
        if (anim != null) anim.cancel();
        super.onDetachedFromWindow();
    }

    @Override
    protected void onDraw(Canvas c) {
        float cx = getWidth() / 2f, cy = getHeight() / 2f, R = Math.min(cx, cy);
        float d = getResources().getDisplayMetrics().density;
        if (camera) {
            p.setStyle(Paint.Style.STROKE);
            p.setStrokeWidth(4 * d);
            p.setColor(0xFFFFFFFF);
            c.drawCircle(cx, cy, R - 3 * d, p);
            p.setStyle(Paint.Style.FILL);
            p.setColor(Ui.PRIMARY);
            if (recording) {
                float h = R * 0.42f;
                r.set(cx - h, cy - h, cx + h, cy + h);
                c.drawRoundRect(r, 6 * d, 6 * d, p);
            } else {
                c.drawCircle(cx, cy, R - 9 * d, p);
            }
            return;
        }
        float ring = R;
        if (progress >= 0) {
            float sw = 10 * d;
            r.set(cx - R + sw / 2, cy - R + sw / 2, cx + R - sw / 2, cy + R - sw / 2);
            p.setStyle(Paint.Style.STROKE);
            p.setStrokeCap(Paint.Cap.ROUND);
            p.setStrokeWidth(sw);
            p.setColor(Ui.LINE);
            c.drawArc(r, 0, 360, false, p);
            if (progress > 0) {
                p.setColor(Ui.PRIMARY);
                c.drawArc(r, -90, 360 * Math.min(1f, progress), false, p);
            }
            ring = R - sw - 12 * d;
        }
        p.setStyle(Paint.Style.FILL);
        if (glow) {   // 퍼지는 빛 두 겹
            for (int k = 0; k < 2; k++) {
                float t = (pulse + k * 0.5f) % 1f;
                p.setColor(Ui.PRIMARY);
                p.setAlpha((int) (70 * (1 - t)));
                c.drawCircle(cx, cy, ring * (0.72f + 0.28f * t), p);
            }
            p.setAlpha(255);
        }
        p.setColor(0xFFFFFFFF);
        c.drawCircle(cx, cy, ring * 0.62f, p);
        p.setColor(Ui.PRIMARY);
        c.drawCircle(cx, cy, ring * 0.54f, p);
    }
}
