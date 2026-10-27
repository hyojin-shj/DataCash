package com.datacash.app;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.Locale;

/** 폰 가속도(가속도 이벤트마다 한 줄) + 가장 최근 자이로 → phone_imu.csv (t_ns 는 센서 BOOTTIME). */
final class Imu implements SensorEventListener {
    private final SensorManager sm;
    private BufferedWriter w;
    private final float[] gyro = new float[3];
    private long rows;

    Imu(Context c) {
        sm = (SensorManager) c.getSystemService(Context.SENSOR_SERVICE);
    }

    void start(File dir) throws Exception {
        w = new BufferedWriter(new FileWriter(new File(dir, "phone_imu.csv")));
        w.write("t_ns,ax,ay,az,gx,gy,gz\n");
        Sensor a = sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER), g = sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        if (g != null) sm.registerListener(this, g, SensorManager.SENSOR_DELAY_FASTEST);
        if (a != null) sm.registerListener(this, a, SensorManager.SENSOR_DELAY_FASTEST);
    }

    @Override
    public synchronized void onSensorChanged(SensorEvent e) {
        if (e.sensor.getType() == Sensor.TYPE_GYROSCOPE) {
            System.arraycopy(e.values, 0, gyro, 0, 3);
            return;
        }
        try {
            w.write(String.format(Locale.US, "%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n", e.timestamp,
                    e.values[0], e.values[1], e.values[2], gyro[0], gyro[1], gyro[2]));
            rows++;
        } catch (Exception ignored) {
        }
    }

    @Override
    public void onAccuracyChanged(Sensor s, int a) {}

    synchronized long stop() {
        sm.unregisterListener(this);
        try { w.close(); } catch (Exception ignored) {}
        return rows;
    }
}
