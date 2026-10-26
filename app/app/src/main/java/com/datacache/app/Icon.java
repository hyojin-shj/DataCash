package com.datacache.app;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import android.view.View;

/** 직접 그리는 선 아이콘 (24 단위 칸 기준, 크기에 맞춰 늘림). 이미지 파일 없이 색만 바꿔 쓴다. */
final class Icon extends View {
    static final int BACK = 0, CHEVRON = 1, STAR = 2, WATCH = 3, FILM = 4, SHIELD = 5, CLOUD = 6, HAND = 7,
            TARGET = 8, CHECK = 9, PERSON = 10, SPARK = 11, LIST = 12, FACE = 13, PLUS = 14, SWITCH = 15;

    private final int kind;
    private final Paint line = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint fill = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Path p = new Path();
    private final RectF r = new RectF();

    Icon(Context c, int kind, int color) {
        super(c);
        this.kind = kind;
        line.setStyle(Paint.Style.STROKE);
        line.setStrokeCap(Paint.Cap.ROUND);
        line.setStrokeJoin(Paint.Join.ROUND);
        line.setColor(color);
        fill.setColor(color);
    }

    @Override
    protected void onDraw(Canvas c) {
        float w = getWidth() - getPaddingLeft() - getPaddingRight(), h = getHeight() - getPaddingTop() - getPaddingBottom();
        float s = Math.min(w, h) / 24f;
        c.save();
        c.translate(getPaddingLeft() + (w - 24 * s) / 2, getPaddingTop() + (h - 24 * s) / 2);
        c.scale(s, s);
        line.setStrokeWidth(2f);
        p.reset();
        switch (kind) {
            case BACK:
                line.setStrokeWidth(2.4f);
                p.moveTo(15, 4); p.lineTo(7, 12); p.lineTo(15, 20);
                c.drawPath(p, line);
                break;
            case CHEVRON:
                p.moveTo(9, 5); p.lineTo(16, 12); p.lineTo(9, 19);
                c.drawPath(p, line);
                break;
            case STAR:
                for (int i = 0; i < 10; i++) {
                    double a = Math.PI / 2 + i * Math.PI / 5;
                    float rad = i % 2 == 0 ? 10f : 4.3f;
                    float x = 12 + (float) (rad * Math.cos(a)), y = 13 - (float) (rad * Math.sin(a));
                    if (i == 0) p.moveTo(x, y); else p.lineTo(x, y);
                }
                p.close();
                c.drawPath(p, fill);
                break;
            case WATCH:
                r.set(6, 6, 18, 18);
                c.drawRoundRect(r, 4, 4, line);
                c.drawLine(9, 6, 10, 2, line); c.drawLine(15, 6, 14, 2, line);
                c.drawLine(9, 18, 10, 22, line); c.drawLine(15, 18, 14, 22, line);
                c.drawLine(12, 12, 12, 9.5f, line); c.drawLine(12, 12, 14, 13, line);
                break;
            case FILM:
                r.set(3, 5, 21, 19);
                c.drawRoundRect(r, 4, 4, line);
                p.moveTo(10, 9); p.lineTo(15.5f, 12); p.lineTo(10, 15); p.close();
                c.drawPath(p, fill);
                break;
            case SHIELD:
                p.moveTo(12, 3); p.lineTo(19.5f, 6); p.lineTo(19.5f, 11.5f);
                p.cubicTo(19.5f, 16, 16, 19.5f, 12, 21); p.cubicTo(8, 19.5f, 4.5f, 16, 4.5f, 11.5f);
                p.lineTo(4.5f, 6); p.close();
                c.drawPath(p, line);
                p.reset();
                p.moveTo(8.8f, 12); p.lineTo(11.2f, 14.4f); p.lineTo(15.5f, 9.8f);
                c.drawPath(p, line);
                break;
            case CLOUD:
                p.moveTo(7, 18); p.cubicTo(3.5f, 18, 2.5f, 13.5f, 6, 12.5f);
                p.cubicTo(6.5f, 8, 12, 6.5f, 14, 10); p.cubicTo(17, 8.5f, 20.5f, 11, 19.5f, 14);
                p.cubicTo(22, 15, 21, 18, 18.5f, 18); p.close();
                c.drawPath(p, line);
                break;
            case HAND:   // 쥐는 손 (주먹 + 엄지)
                r.set(5, 8, 17, 19);
                c.drawRoundRect(r, 4, 4, line);
                c.drawLine(9, 8, 9, 12, line); c.drawLine(13, 8, 13, 12, line);
                p.moveTo(17, 11); p.cubicTo(20, 11, 20, 15, 17, 15);
                c.drawPath(p, line);
                c.drawLine(8, 19, 8, 22, line); c.drawLine(14, 19, 14, 22, line);
                break;
            case TARGET:
                c.drawCircle(12, 12, 8.5f, line);
                c.drawCircle(12, 12, 4.5f, line);
                c.drawCircle(12, 12, 1.6f, fill);
                c.drawLine(17, 7, 21, 3, line);
                break;
            case CHECK:
                line.setStrokeWidth(2.6f);
                p.moveTo(5, 12.5f); p.lineTo(10, 17); p.lineTo(19, 7);
                c.drawPath(p, line);
                break;
            case PERSON:
                c.drawCircle(12, 8, 3.8f, line);
                p.moveTo(4.5f, 20); p.cubicTo(5.5f, 14.5f, 18.5f, 14.5f, 19.5f, 20);
                c.drawPath(p, line);
                break;
            case SPARK:   // 4감 (작은 별 둘)
                p.moveTo(10, 3); p.lineTo(11.5f, 8.5f); p.lineTo(17, 10); p.lineTo(11.5f, 11.5f);
                p.lineTo(10, 17); p.lineTo(8.5f, 11.5f); p.lineTo(3, 10); p.lineTo(8.5f, 8.5f); p.close();
                c.drawPath(p, fill);
                c.drawCircle(18, 18, 2.2f, fill);
                break;
            case LIST:
                for (int k = 0; k < 3; k++) {
                    c.drawCircle(5, 6 + 6 * k, 1.4f, fill);
                    c.drawLine(9, 6 + 6 * k, 20, 6 + 6 * k, line);
                }
                break;
            case FACE:   // 얼굴 확인 (네 모서리 + 얼굴)
                p.moveTo(3, 8); p.lineTo(3, 3); p.lineTo(8, 3);
                p.moveTo(16, 3); p.lineTo(21, 3); p.lineTo(21, 8);
                p.moveTo(21, 16); p.lineTo(21, 21); p.lineTo(16, 21);
                p.moveTo(8, 21); p.lineTo(3, 21); p.lineTo(3, 16);
                c.drawPath(p, line);
                c.drawCircle(9.5f, 10.5f, 1.1f, fill);
                c.drawCircle(14.5f, 10.5f, 1.1f, fill);
                p.reset();
                p.moveTo(9, 15); p.cubicTo(10.5f, 16.8f, 13.5f, 16.8f, 15, 15);
                c.drawPath(p, line);
                break;
            case SWITCH:   // 앞뒤 카메라 (돌아가는 두 화살표)
                r.set(4, 4, 20, 20);
                c.drawArc(r, 200, 140, false, line);
                c.drawArc(r, 20, 140, false, line);
                p.moveTo(19.5f, 4.5f); p.lineTo(19.8f, 9); p.lineTo(15.4f, 8.6f);
                p.moveTo(4.5f, 19.5f); p.lineTo(4.2f, 15); p.lineTo(8.6f, 15.4f);
                c.drawPath(p, line);
                break;
            case PLUS:
                line.setStrokeWidth(2.4f);
                c.drawLine(12, 5, 12, 19, line);
                c.drawLine(5, 12, 19, 12, line);
                break;
            default:
                break;
        }
        c.restore();
    }
}
