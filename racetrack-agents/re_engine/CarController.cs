// ============================================================================
// CarController.cs — drives a car from the policy's (acceleration, steering)
// commands using the EXACT dynamics the model was trained on: highway-env's
// BicycleVehicle (linear tire model with slip, RK4-integrated; Rajamani,
// "Vehicle Dynamics and Control", ch. 2). Same trajectories as the sim.
//
// The engine owns the Transform; this class owns the physics state and
// GIVES OUT the world velocity as a Vector3 every step. Two ways to use it:
//
//   var car = new CarController();
//   car.SetPose(tf.position.x, tf.position.z,           // sync once at spawn
//               HeadingFromForward(tf.forward), speed: 0);
//
//   // every physics tick (any rate; 1/60 s works — internally exact RK4):
//   Vector3 vel = car.Step(dt, accel, steerRad);         // m/s, (X, 0, Z)
//
//   //  A) move the transform yourself with the returned velocity:
//   tf.position += vel * dt;                             // or rb.velocity = vel
//   //  B) or copy the integrated pose directly:
//   tf.position = new Vector3((float)car.X, tf.position.y, (float)car.Y);
//   yawDegrees  = 90.0 - car.Heading * 180.0 / Math.PI;  // engine Euler yaw
//
// Pick ONE of A/B — mixing both double-integrates. Units are meters and
// m/s everywhere (multiply the returned velocity by 3.6f for km/h).
// Commands come from ActionDecoder (+ SteeringSmoother / CorneringLimiter /
// StuckRecovery); hold the last command between 5 Hz policy ticks and keep
// calling Step every frame so the physics stay smooth.
// ============================================================================

using System;
using System.Numerics;

namespace RacetrackSingle
{
    public sealed class CarController
    {
        // -------- constants copied from highway_env BicycleVehicle --------
        // NOTE: these derive from the CLASS defaults (LENGTH 5, WIDTH 2),
        // not from car_length in race_env config — highway-env computes them
        // once at class level, so the trained dynamics really used these.
        const double Mass = 1.0;                     // [kg]
        const double LengthA = 2.5;                  // [m] CG -> front axle
        const double LengthB = 2.5;                  // [m] CG -> rear axle
        const double InertiaZ = (25.0 + 4.0) / 12.0; // [kg m^2] 1/12 (L^2+W^2)
        const double FrictionFront = 15.0 * Mass;    // [N]
        const double FrictionRear = 15.0 * Mass;     // [N]
        const double MaxAngularSpeed = 2.0 * Math.PI;   // [rad/s]
        const double MaxSteerPhysical = Math.PI / 2.0;  // model linearisation limit

        // training config values (ego_min_speed / ego_max_speed)
        public double MinSpeed = -8.0;               // [m/s]
        public double MaxSpeed = 16.0;               // [m/s]

        // -------- state (track plane: X = world X, Y = world Z) --------
        public double X { get; private set; }
        public double Y { get; private set; }
        public double Heading { get; private set; }       // [rad] CCW from +X
        public double Speed { get; private set; }         // [m/s] longitudinal (body)
        public double LateralSpeed { get; private set; }  // [m/s] body-lateral (slip)
        public double YawRate { get; private set; }       // [rad/s]

        // World velocity of the LAST Step call, ground plane in (X, 0, Z), m/s.
        public Vector3 Velocity { get; private set; }

        // Sync the physics state to the engine transform (spawn / respawn /
        // teleport). heading in RADIANS (use the Heading helpers to convert).
        public void SetPose(double x, double y, double heading, double speed = 0.0)
        {
            X = x; Y = y; Heading = heading; Speed = speed;
            LateralSpeed = 0.0; YawRate = 0.0;
            Velocity = ToWorldVelocity();
        }

        // Advance the car by dt with the commanded acceleration [m/s^2] and
        // front-wheel steering angle [rad]. Returns the world velocity Vector3.
        public Vector3 Step(double dt, double acceleration, double steering)
        {
            // clip_actions: speed saturation via acceleration, exactly as trained
            if (Speed > MaxSpeed)
                acceleration = Math.Min(acceleration, MaxSpeed - Speed);
            else if (Speed < MinSpeed)
                acceleration = Math.Max(acceleration, MinSpeed - Speed);
            if (steering > MaxSteerPhysical) steering = MaxSteerPhysical;
            if (steering < -MaxSteerPhysical) steering = -MaxSteerPhysical;
            if (YawRate > MaxAngularSpeed) YawRate = MaxAngularSpeed;
            if (YawRate < -MaxAngularSpeed) YawRate = -MaxAngularSpeed;

            // RK4 over state [x, y, heading, speed, lateralSpeed, yawRate]
            double[] s0 = { X, Y, Heading, Speed, LateralSpeed, YawRate };
            double[] f1 = Derivative(s0, acceleration, steering);
            double[] f2 = Derivative(AddScaled(s0, f1, dt / 2.0), acceleration, steering);
            double[] f3 = Derivative(AddScaled(s0, f2, dt / 2.0), acceleration, steering);
            double[] f4 = Derivative(AddScaled(s0, f3, dt), acceleration, steering);
            for (int i = 0; i < 6; i++)
                s0[i] += dt / 6.0 * (f1[i] + 2.0 * f2[i] + 2.0 * f3[i] + f4[i]);

            X = s0[0]; Y = s0[1]; Heading = s0[2];
            Speed = s0[3]; LateralSpeed = s0[4]; YawRate = s0[5];
            Velocity = ToWorldVelocity();
            return Velocity;
        }

        // Rajamani ch.2 lateral dynamics with highway-env's low-speed damping.
        static double[] Derivative(double[] s, double acceleration, double steering)
        {
            double heading = s[2], speed = s[3], lat = s[4], yaw = s[5];

            double thetaVf = Math.Atan2(lat + LengthA * yaw, speed);   // (2.27)
            double thetaVr = Math.Atan2(lat - LengthB * yaw, speed);   // (2.28)
            double fyf = 2.0 * FrictionFront * (steering - thetaVf);   // (2.25)
            double fyr = 2.0 * FrictionRear * (0.0 - thetaVr);         // (2.26)
            if (Math.Abs(speed) < 1.0)   // low speed: damp lateral speed + yaw
            {
                fyf = -Mass * lat - InertiaZ / LengthA * yaw;
                fyr = -Mass * lat + InertiaZ / LengthA * yaw;
            }
            double dLat = (fyf + fyr) / Mass - yaw * speed;            // (2.21)
            double dYaw = (LengthA * fyf - LengthB * fyr) / InertiaZ;  // (2.22)

            double c = Math.Cos(heading), sn = Math.Sin(heading);
            return new[]
            {
                c * speed - sn * lat,   // dx
                sn * speed + c * lat,   // dy
                yaw,                    // dheading
                acceleration,           // dspeed
                dLat,
                dYaw,
            };
        }

        static double[] AddScaled(double[] a, double[] b, double k)
        {
            var r = new double[6];
            for (int i = 0; i < 6; i++) r[i] = a[i] + b[i] * k;
            return r;
        }

        Vector3 ToWorldVelocity()
        {
            double c = Math.Cos(Heading), sn = Math.Sin(Heading);
            return new Vector3(
                (float)(c * Speed - sn * LateralSpeed),
                0f,
                (float)(sn * Speed + c * LateralSpeed));
        }
    }
}
