import numpy as np
import numpy.random
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd


def comp_plot_train(data, columns, names, x_title, y_title, titles, uncertainty_cols):
    df = []
    for d in data:
        df.append(pd.DataFrame(d))

    graph = []

    for _ in columns:
        graph.append(go.Figure())

    for j, d in enumerate(df):
        for i, col in enumerate(columns):
            if uncertainty_cols[i] is None:
                graph[i].add_trace(go.Scatter(
                    x=d['t'],
                    y=d[col],
                    mode='lines',
                    name=names[j], )
                )
            else:
                graph[i].add_trace(go.Scatter(
                    x=d['t'],
                    y=d[col],
                    mode='lines',
                    name=names[j],
                    error_y=dict(
                        type='data',
                        array=d[uncertainty_cols[i]],
                        visible=True
                    )),
                )

    for i, g in enumerate(graph):
        g.update_layout(title=titles[i],
                        xaxis_title=x_title[i],
                        yaxis_title=y_title[i])
        g.show()


class Train:
    def __init__(self, mu, sigma, C, motion_noise, obs_noise):
        self.mu = mu
        self.sigma = sigma
        self.C = C
        self.true_X = np.array([0, 0])
        self.motion_noise = np.sqrt(motion_noise)
        self.obs_noise = np.sqrt(obs_noise)

    def get_ut(self, t):
        if t < 0.25:
            return np.array([400*0.01])
        elif 3 < t < 3.25:
            return np.array([-400*0.01])
        else:
            return np.array([0])

    def update_true(self, t):
        noise_pos = np.random.normal(0, self.motion_noise[0])
        noise_vel = np.random.normal(0, self.motion_noise[1])
        if t < 0.25:
            return self.true_X + np.array([self.true_X[1] * 0.01, 0.01*400]) + np.array([noise_pos, noise_vel])
        elif 3 < t < 3.25:
            return self.true_X + np.array([self.true_X[1] * 0.01, -400*0.01]) + np.array([noise_pos, noise_vel])
        else:
            return self.true_X + np.array([self.true_X[1] * 0.01, 0]) + np.array([noise_pos, noise_vel])

    def get_zt(self):
        return np.matmul(self.C, self.true_X.reshape((2, 1))) + np.random.normal(0, self.obs_noise[0])

    def plot(self, estimate_x):

        df = pd.DataFrame(estimate_x)

        true_pos_vel_graph = make_subplots(rows=2, cols=1)

        true_pos_vel_graph.add_trace(go.Scatter(
            x=df['t'],
            y=df['position'],
            mode='lines',
            name='Position',), row=1,col=1
        )

        true_pos_vel_graph.add_trace(go.Scatter(
            x=df['t'],
            y=df['velocity'],
            mode='lines',
            name='Velocity', ), row=2, col=1
        )

        true_pos_vel_graph.update_xaxes(title_text="Time", row=1, col=1)
        true_pos_vel_graph.update_xaxes(title_text="Time", row=2, col=1)

        true_pos_vel_graph.update_yaxes(title_text="Distance from A", row=1, col=1)
        true_pos_vel_graph.update_yaxes(title_text="Velocity", row=2, col=1)
        true_pos_vel_graph.update_layout(title_text="True Position")
        true_pos_vel_graph.show()

        est_position_graph = go.Figure()
        est_position_graph.add_trace(go.Scatter(
            x=df['t'],
            y=df['position_mean'],
            mode='lines',
            name='Estimated Position',
            error_y=dict(
                type='data',
                array=df['position_var'],
                visible=True
            )),
        )

        est_position_graph.update_layout(
            title='Estimated Position of Train vs Time',
            xaxis_title='Time',
            yaxis_title='Distance from A'
        )

        est_position_graph.show()

        est_position_vs_true_graph = go.Figure()
        est_position_vs_true_graph.add_trace(go.Scatter(
            x=df['t'],
            y=df['position_mean'],
            mode='lines',
            name='Estimated Position',
            error_y=dict(
                type='data',
                array=df['position_var'],
                visible=True
            )),
        )

        est_position_vs_true_graph.add_trace(go.Scatter(
            x=df['t'],
            y=df['position'],
            mode='lines',
            name='True position',
        ),)

        est_position_vs_true_graph.update_layout(
            title='True and Estimated Position of Train vs Time',
            xaxis_title='Time',
            yaxis_title='Distance from A'
        )

        velocity_graph = go.Figure()
        velocity_graph.add_trace(go.Scatter(
            x=df['t'],
            y=df['velocity_mean'],
            mode='lines',
            name='Estimated_Velocity',
            error_y=dict(
                type='data',
                array=df['velocity_var'],
                visible=True
            )

        ))

        velocity_graph.update_layout(
            title='Velocity of Train vs Time',
            xaxis_title='Time',
            yaxis_title='Velocity of Train (Km/h)'
        )

        kalman_gain_graph = go.Figure()

        kalman_gain_graph.add_trace(go.Scatter(
            x=df['t'],
            y=df['kalman_gain_x'],
            mode='lines',
            name='Kalman Gain X',
        ))


        kalman_gain_graph.add_trace(go.Scatter(
            x=df['t'],
            y=df['kalman_gain_x_dot'],
            mode='lines',
            name='Kalman Gain X Dot',
        ))

        kalman_gain_graph.update_layout(
            title='Kalman Gain vs Time for Train',
            xaxis_title='Time',
            yaxis_title='Kalman Gain'
        )

        est_position_vs_true_graph.show()
        velocity_graph.show()
        kalman_gain_graph.show()


def run_train(t_start=None, t_end=None, custom_motion_noise=None, custom_obs_noise=None,plot=True):
    A = np.array([[1, 0.01],
                  [0, 1]])

    B = np.array([[0],
                  [1]])

    if custom_motion_noise is None:
        motion_noise = [0.1 ** 2, 0.5 ** 2]
    else:
        motion_noise = custom_motion_noise

    if custom_obs_noise is None:
        obs_noise = [0.01 ** 2]
    else:
        obs_noise = motion_noise

    R = np.array([[motion_noise[0], 0],
                  [0, motion_noise[1]]])

    Q = np.array([obs_noise[0]])

    C = np.array([[1 / 1500, 0]]).reshape((1, 2))

    init_mu = np.random.normal(0, 10e-2, 2).reshape((2, 1))
    init_sigma = np.array([[10e-4, 0], [0, 10e-4]])
    train = Train(init_mu, init_sigma, C, motion_noise, obs_noise)
    Kfilter = KalmanFilter(A, B, R, Q, C, "train")

    for t in np.arange(0, 3.27, 0.01):
        ut = train.get_ut(t)
        if t_start is not None:
            if t_start <= t <= t_end:
                mu, sigma = Kfilter.update(train.mu, train.sigma, ut, None, t, train.true_X)
            else:
                zt = train.get_zt()
                mu, sigma = Kfilter.update(train.mu, train.sigma, ut, zt, t, train.true_X)

        else:
            zt = train.get_zt()
            mu, sigma = Kfilter.update(train.mu, train.sigma, ut, zt, t, train.true_X)

        train.mu = mu
        train.sigma = sigma
        train.true_X = train.update_true(t)

    if plot:
        train.plot(Kfilter.estimated_x)

    return train, Kfilter


class KalmanFilter:

    def __init__(self, A, B, R, Q, C, mode, n=2):

        self.A = A
        self.B = B
        self.K = np.array([[0, 0]]).reshape((2, 1))
        self.Q = Q
        self.C = C
        self.R = R
        self.mode = mode
        self.n = n
        self.estimated_x = []

    def update(self, mu, sigma, u_t, z_t, t, true_X):

        if self.mode == "train":
            self.estimated_x.append({"position": true_X[0], "velocity": true_X[1],
                                     "position_mean":  mu[0][0], "velocity_mean": mu[1][0],
                                     "position_var": np.sqrt(sigma[0][0]),
                                     "velocity_var": np.sqrt(sigma[1][1]), "kalman_gain_x": self.K[0][0],
                                     "kalman_gain_x_dot": self.K[1][0],  "t": t})

        elif self.mode == "football":
            self.estimated_x.append({"x_g": true_X[0][0], "y_g": true_X[1][0], "z_g": true_X[2][0],
                                     "xdot_g": true_X[3][0], "ydot_g": true_X[4][0], "zdot_g": true_X[5][0],
                                     "x": mu[0][0], "y": mu[1][0], "z": mu[2][0],
                                     "xdot": mu[3][0], "ydot": mu[4][0], "zdot": mu[5][0],
                                     "x_var": sigma[0][0], "y_var": sigma[1][1], "z_var": sigma[2][2],
                                     "xdot_var": sigma[3][3], "ydot_var": sigma[4][4], "zdot_var": sigma[5][5],
                                     "kalman_gain": self.K, "t": t
                                     })
        #temp = np.matmul(self.B, u_t).reshape(self.n,1)
        mu_prediction = np.add(np.matmul(self.A, mu), np.matmul(self.B, u_t).reshape(self.n, 1))

        sigma_prediction = np.add(np.matmul(np.matmul(self.A, sigma), self.A.T), self.R)

        temp = np.add(np.matmul(np.matmul(self.C, sigma_prediction), self.C.T), self.Q)
        if temp.shape == (1, 1):
            temp = 1/temp
        else:
            temp = np.linalg.inv(temp)

        if z_t is not None:
            self.K = np.matmul(np.matmul(sigma_prediction, self.C.T), temp)
            mu_estimated = np.add(mu_prediction, np.matmul(self.K, np.subtract(z_t, np.matmul(self.C, mu_prediction))))
            sigma_estimated = np.matmul(np.subtract(np.identity(self.n),
                                             np.matmul(self.K, self.C)), sigma_prediction)

            return mu_estimated, sigma_estimated
        else:

            return mu_prediction, sigma_prediction


class Football:
    def __init__(self, mu, sigma, C, non_linear_obs=False, state_noise=None, observation_noise=None, obs_space=3, field_2d=False, field_mode=0):
        self.mu = mu
        self.sigma = sigma
        self.C = C
        self.field_2d = field_2d
        self.field_mode = field_mode

        if not field_2d:
            self.true_X = np.array([[24.0, 4.0, 0.0, -16.04, 36.8, 8.61]]).reshape((6,1))
        else:
            self.true_X = np.array([[0.0, 50.0, 0.0, 40.0]]).reshape((4, 1))

        self.non_linear_obs = non_linear_obs
        self.state_noise = np.sqrt(state_noise)
        self.observation_noise = np.sqrt(observation_noise)
        self.obs_space = obs_space

    def get_ut(self):
        if not self.field_2d:
            return np.array([[-10*0.01]])

    def update_true(self):
        if self.field_2d:
            noise_pos = np.random.normal(0, self.state_noise[0], (2, 1))
            noise_vel = np.random.normal(0, self.state_noise[3], (2, 1))

            noise = np.concatenate((noise_pos, noise_vel), axis=0)
            xdot = self.true_X[2][0]
            ydot = self.true_X[3][0]
            temp = np.array([xdot * 0.01, ydot * 0.01, 0, 0]).reshape((4, 1))
            return self.true_X + temp + noise.reshape((4, 1))

        else:
            noise_pos = np.random.normal(0, self.state_noise[0], (3, 1))
            noise_vel = np.random.normal(0, self.state_noise[3], (3, 1))

            noise = np.concatenate((noise_pos, noise_vel), axis=0)
            xdot = self.true_X[3][0]
            ydot = self.true_X[4][0]
            zdot = self.true_X[5][0]
            temp = np.array([xdot*0.01, ydot*0.01, zdot*0.01, 0, 0, -10*0.01]).reshape((6, 1))
            return self.true_X + temp + noise.reshape((6, 1))

    def get_zt(self):
        if self.obs_space == 3:

            if not self.non_linear_obs:
                return np.matmul(self.C, self.true_X.reshape((6, 1))) + np.random.normal(0, self.observation_noise[0], ((3,1)))
            else:
                obs_noise = self.observation_noise

                if not self.field_2d:

                    true_x = [self.true_X[0][0], self.true_X[1][0], self.true_X[2][0]]
                    bs1 = [32, -50, 10]
                    bs2 = [-32, -50, 10]
                    bs3 = [-32, 50, 10]
                    bs4 = [32, 50, 10]

                    d = np.array([[np.sqrt((bs1[0] - true_x[0]) ** 2 + (bs1[1] - true_x[1]) ** 2 + (
                                bs1[2] - true_x[2]) ** 2) + np.random.normal(0, obs_noise[0]),
                         np.sqrt((bs2[0] - true_x[0]) ** 2 + (bs2[1] - true_x[1]) ** 2 + (
                                     bs2[2] - true_x[2]) ** 2) + np.random.normal(0, obs_noise[1]),
                         np.sqrt((bs3[0] - true_x[0]) ** 2 + (bs3[1] - true_x[1]) ** 2 + (
                                     bs3[2] - true_x[2]) ** 2) + np.random.normal(0, obs_noise[2]),
                         np.sqrt((bs4[0] - true_x[0]) ** 2 + (bs4[1] - true_x[1]) ** 2 + (
                                     bs4[2] - true_x[2]) ** 2) + np.random.normal(0, obs_noise[3])]]).reshape((4, 1))

                    return d
                else:
                    true_x = [self.true_X[0][0], self.true_X[1][0]]
                    if self.field_mode == 0:
                        bs3 = [-32, 50]
                        bs4 = [32, 50]

                        d = np.array([[np.sqrt((bs3[0] - true_x[0]) ** 2 + (bs3[1] - true_x[1]) ** 2) + np.random.normal(0, obs_noise[0]),
                             np.sqrt((bs4[0] - true_x[0]) ** 2 + (bs4[1] - true_x[1]) ** 2) + np.random.normal(0, obs_noise[1])]]).reshape((2,1))

                    if self.field_mode == 1:
                        bs1 = [32, -50]
                        bs4 = [32, 50]

                        d = np.array([[np.sqrt((bs1[0] - true_x[0]) ** 2 + (bs1[1] - true_x[1]) ** 2) + np.random.normal(0, obs_noise[0]),
                             np.sqrt((bs4[0] - true_x[0]) ** 2 + (bs4[1] - true_x[1]) ** 2) + np.random.normal(0, obs_noise[1])]]).reshape((2,1))

                    if self.field_mode == 2:
                        bs2 = [-32, -50]
                        bs4 = [32, 50]

                        d = np.array([[
                             np.sqrt((bs2[0] - true_x[0]) ** 2 + (bs2[1] - true_x[1]) ** 2) + np.random.normal(0, obs_noise[0]),
                             np.sqrt((bs4[0] - true_x[0]) ** 2 + (bs4[1] - true_x[1]) ** 2) + np.random.normal(0, obs_noise[1])]]).reshape((2,1))
                    return d
        else:
            return np.matmul(self.C, self.true_X.reshape((6, 1))) + np.random.normal(0, self.observation_noise[0], ((6, 1)))

    def plot(self, estimate_X, title=""):
        fig = go.Figure()
        field = [
            [32, -50, 0],
            [-32, -50, 0],
            [-32, 50, 0],
            [32, 50, 0],
            [32, -50, 0]
        ]
        goal = [
            [4, 50, 3],
            [-4, 50, 3],
            [-4, 50, 0],
            [4, 50, 0],
            [4, 50, 3]
        ]
        field_x, field_y, field_z = zip(*field)
        goal_x, goal_y, goal_z = zip(*goal)

        #drawing field
        fig.add_trace(go.Scatter3d(
            x=field_x, y=field_y, z=field_z,
            mode='lines',
            name='Playing Field',
            line=dict(color='red', width=2)
        ))
        #drawing goal post
        fig.add_trace(go.Scatter3d(
            x=goal_x, y=goal_y, z=goal_z,
            mode='lines',
            name='Goal Post',
            line=dict(color='blue', width=2)
        ))

        #drawing estimated trajectory
        df = pd.DataFrame(estimate_X)
        fig.add_trace(go.Scatter3d(
            x=df["x"],
            y=df["y"],
            z=df["z"],
            name='Estimated Position',
            mode='lines',
        ))
        # drawing ground truth trajectory
        fig.add_trace(go.Scatter3d(
            x=df["x_g"],
            y=df["y_g"],
            z=df["z_g"],
            name='True Position',
            mode='lines',
        ))

        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[-60, 60], dtick=10),
                yaxis=dict(range=[-60, 60], dtick=10),
                zaxis=dict(range=[-60, 60], dtick=10)
            )
        )

        fig.update_layout(
            title='FootBall Trajectory ' + title,
            scene=dict(
                xaxis_title='X Axis',
                yaxis_title='Y Axis',
                zaxis_title='Z Axis'
            )
        )

        fig.show()


def run_football_gps(traj_precal= None, custom_state_noise=None, custom_observation_noise= None, title="Given_Noise_With_GPS"):
    numpy.random.seed(653)
    A = np.array([[1, 0, 0, 0.01, 0, 0],
                  [0, 1, 0, 0, 0.01, 0],
                  [0, 0, 1, 0, 0, 0.01],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])

    B = np.array([[0], [0], [0], [0], [0], [1]]).reshape((6, 1))

    if custom_state_noise is None:
        r = [0.01**2, 0.01**2, 0.01**2, 0.1**2, 0.1**2, 0.1**2]

    else:
        r = custom_state_noise
    custom_state_noise = r

    R = np.array([[r[0], 0, 0, 0, 0, 0],
                  [0, r[1], 0, 0, 0, 0],
                  [0, 0, r[2], 0, 0, 0],
                  [0, 0, 0, r[3], 0, 0],
                  [0, 0, 0, 0, r[4], 0],
                  [0, 0, 0, 0, 0, r[5]]])

    if custom_observation_noise is None:
        q = [0.1**2, 0.1**2, 0.1**2]
    else:
        q = custom_observation_noise

    custom_observation_noise = q

    Q = np.array([[q[0], 0, 0],
                  [0, q[1], 0],
                  [0, 0, q[2]]])

    C = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0]])

    init_state = np.add(np.array([24.0, 4.0, 0.0, -16.04, 36.8, 8.61]), numpy.random.normal(0, 10e-2, size=(6))).reshape((6,1))
    var = np.array([[0.0001, 0, 0, 0, 0, 0],
                  [0, 0.0001, 0, 0, 0, 0],
                  [0, 0, 0.0001, 0, 0, 0],
                  [0, 0, 0, 0.0001, 0, 0],
                  [0, 0, 0, 0, 0.0001, 0],
                  [0, 0, 0, 0, 0, 0.0001]])

    if traj_precal is None:
        football = Football(init_state, var, C, non_linear_obs=False, state_noise=custom_state_noise, observation_noise=custom_observation_noise)
    else:
        football = Football(traj_precal[0], var, C, non_linear_obs=False, state_noise=custom_state_noise, observation_noise=custom_observation_noise)

    Kfilter = KalmanFilter(A, B, R, Q, C, "football", 6)

    i = 1
    sensor_traj = []
    for t in np.arange(0, 1.30, 0.01):
        ut = football.get_ut()
        zt = football.get_zt()
        sensor_traj.append({"x_s": zt[0][0], "y_s": zt[1][0]})
        mu, sigma = Kfilter.update(football.mu, football.sigma, ut, zt, t, football.true_X)
        football.mu = mu
        football.sigma = sigma
        if traj_precal is None:
            football.true_X = football.update_true()
        else:
            football.true_X = traj_precal[i]
            i += 1

    football.plot(Kfilter.estimated_x, title)
    plot_XY_projection(Kfilter.estimated_x, sensor_traj, title)
    return Kfilter


def run_football_imu(traj_precal=None, custom_state_noise = None, custom_observation_noise=None,title="Given_Noise_With_IMU"):

    A = np.array([[1, 0, 0, 0.01, 0, 0],
                  [0, 1, 0, 0, 0.01, 0],
                  [0, 0, 1, 0, 0, 0.01],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])

    B = np.array([[0], [0], [0], [0], [0], [1]]).reshape((6, 1))

    if custom_state_noise is None:
        r = [0.01**2, 0.01**2, 0.01**2, 0.1**2, 0.1**2, 0.1**2]
    else:
        r = custom_state_noise
    custom_state_noise = r

    R = np.array([[r[0], 0, 0, 0, 0, 0],
                  [0, r[1], 0, 0, 0, 0],
                  [0, 0, r[2], 0, 0, 0],
                  [0, 0, 0, r[3], 0, 0],
                  [0, 0, 0, 0, r[4], 0],
                  [0, 0, 0, 0, 0, r[5]]])

    if custom_observation_noise is None:
        q = [0.1**2, 0.1**2, 0.1**2, 0.1**2, 0.1**2, 0.1**2]
    else:
        q = custom_observation_noise
    custom_observation_noise = q

    Q = np.array([[q[0], 0, 0, 0, 0, 0],
                  [0, q[1], 0, 0, 0, 0],
                  [0, 0, q[2], 0, 0, 0],
                  [0, 0, 0, q[3], 0, 0],
                  [0, 0, 0, 0, q[4], 0],
                  [0, 0, 0, 0, 0, q[5]]])

    C = np.array([[1, 0, 0.01, 0, 0, 0],
                  [0, 1, 0, 0, 0.01, 0],
                  [0, 0, 1, 0, 0, 0.01],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])

    init_state = np.add(np.array([24.0, 4.0, 0.0, -16.04, 36.8, 8.61]),
                        np.random.normal(0, 10e-2, size=(6))).reshape((6, 1))
    var = np.array([[0.0001, 0, 0, 0, 0, 0],
                    [0, 0.0001, 0, 0, 0, 0],
                    [0, 0, 0.0001, 0, 0, 0],
                    [0, 0, 0, 0.0001, 0, 0],
                    [0, 0, 0, 0, 0.0001, 0],
                    [0, 0, 0, 0, 0, 0.0001]])

    if traj_precal is None:
        football = Football(init_state, var, C, False, custom_state_noise, custom_observation_noise,6)
    else:
        football = Football(traj_precal[0], var, C, False, custom_state_noise, custom_observation_noise,6)

    Kfilter = KalmanFilter(A, B, R, Q, C, "football", 6)

    i = 1
    for t in np.arange(0, 1.30, 0.01):
        ut = football.get_ut()
        zt = football.get_zt()
        mu, sigma = Kfilter.update(football.mu, football.sigma, ut, zt, t, football.true_X)
        football.mu = mu
        football.sigma = sigma
        if traj_precal is None:
            football.true_X = football.update_true()
        else:
            football.true_X = traj_precal[i]
            i += 1

    football.plot(Kfilter.estimated_x,title)
    return Kfilter


class ExtendedKalmanFilter:

    def __init__(self, A, B, R, Q, n=4, bs_mode=None):
        self.A = A
        self.B = B
        self.K = np.array([[0, 0]]).reshape((2, 1))
        self.Q = Q
        self.R = R
        self.n = n
        self.bs_mode = bs_mode
        self.estimated_x = []

    def g(self, mu, u_t):
        if self.n == 6:
            return np.add(np.matmul(self.A, mu), np.matmul(self.B, u_t).reshape(self.n, 1))
        else:
            return np.matmul(self.A, mu).reshape((self.n, 1))

    def G(self):
        return self.A

    def h(self, mu):
        mu = [mu[0][0], mu[1][0], mu[2][0]]

        if self.bs_mode is None:
            bs1 = [32, -50, 10]
            bs2 = [-32, -50, 10]
            bs3 = [-32, 50, 10]
            bs4 = [32, 50, 10]

            d = np.array([[np.sqrt((bs1[0] - mu[0]) ** 2 + (bs1[1] - mu[1]) ** 2 + (bs1[2] - mu[2]) ** 2),
                 np.sqrt((bs2[0] - mu[0]) ** 2 + (bs2[1] - mu[1]) ** 2 + (bs2[2] - mu[2]) ** 2),
                 np.sqrt((bs3[0] - mu[0]) ** 2 + (bs3[1] - mu[1]) ** 2 + (bs3[2] - mu[2]) ** 2),
                 np.sqrt((bs4[0] - mu[0]) ** 2 + (bs4[1] - mu[1]) ** 2 + (bs4[2] - mu[2]) ** 2), ]]).reshape((4,1))
            return d

        else:
            if self.bs_mode == 0:
                bs3 = [-32, 50]
                bs4 = [32, 50]

                d = np.array([[np.sqrt((bs3[0] - mu[0]) ** 2 + (bs3[1] - mu[1]) ** 2 ),
                     np.sqrt((bs4[0] - mu[0]) ** 2 + (bs4[1] - mu[1]) ** 2 )]]).reshape((2,1))

            if self.bs_mode == 1:
                bs1 = [32, -50]
                bs4 = [32, 50]

                d = np.array([[np.sqrt((bs1[0] - mu[0]) ** 2 + (bs1[1] - mu[1]) ** 2),
                     np.sqrt((bs4[0] - mu[0]) ** 2 + (bs4[1] - mu[1]) ** 2) ]]).reshape((2,1))

            if self.bs_mode == 2:
                bs2 = [-32, -50]
                bs4 = [32, 50]

                d = np.array([[np.sqrt((bs2[0] - mu[0]) ** 2 + (bs2[1] - mu[1]) ** 2 ),
                     np.sqrt((bs4[0] - mu[0]) ** 2 + (bs4[1] - mu[1]) ** 2) ]]).reshape((2,1))

            return d

    def H(self, mu, d):
        if self.bs_mode is None:
            return np.array([[2*(mu[0][0] - 32)/d[0][0], 2*(mu[0][0] + 32)/d[1][0], 2*(mu[0][0] + 32)/d[2][0], 2*(mu[0][0] - 32)/d[3][0]],
                              [2*(mu[1][0] + 50)/d[0][0], 2*(mu[1][0] + 50)/d[1][0], 2*(mu[1][0] - 50)/d[2][0], 2*(mu[1][0] - 50)/d[3][0]],
                              [2*(mu[2][0] - 10)/d[0][0], 2*(mu[2][0] - 10)/d[1][0], 2*(mu[2][0] - 10)/d[2][0], 2*(mu[2][0] - 10)/d[3][0]],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]])
        else:
            if self.bs_mode == 0:
                return np.array([[2 * (mu[0][0] + 32) / d[0][0], 2 * (mu[0][0] - 32) / d[1][0]],
                                 [2 * (mu[1][0] - 50) / d[0][0], 2 * (mu[1][0] - 50) / d[1][0]],
                                 [2 * (mu[2][0] - 10) / d[0][0], 2 * (mu[2][0] - 10) / d[1][0]],
                                 [0, 0,],])
            if self.bs_mode == 1:
                return np.array([[2 * (mu[0][0] - 32) / d[0][0], 2 * (mu[0][0] - 32) / d[1][0]],
                                 [2 * (mu[1][0] + 50) / d[0][0], 2 * (mu[1][0] - 50) / d[1][0]],
                                 [2 * (mu[2][0] - 10) / d[0][0], 2 * (mu[2][0] - 10) / d[1][0]],
                                 [0, 0,],
                                 ])
            if self.bs_mode == 2:
                return np.array([[2 * (mu[0][0] + 32) / d[0][0], 2 * (mu[0][0] - 32) / d[1][0]],
                                 [2 * (mu[1][0] + 50) / d[0][0], 2 * (mu[1][0] - 50) / d[1][0]],
                                 [2 * (mu[2][0] - 10) / d[0][0], 2 * (mu[2][0] - 10) / d[1][0]],
                                 [0, 0,],
                                 ])

    def update(self, mu, sigma, u_t, z_t, t, true_X):
        if self.n == 6:
            self.estimated_x.append({"x_g": true_X[0][0], "y_g": true_X[1][0], "z_g": true_X[2][0],
                                     "xdot_g": true_X[3][0], "ydot_g":true_X[4][0], "zdot_g": true_X[5][0],
                                     "x": mu[0][0], "y": mu[1][0], "z": mu[2][0],
                                     "xdot": mu[3][0], "ydot": mu[4][0], "zdot": mu[5][0],
                                     "x_var": sigma[0][0], "y_var": sigma[1][1], "z_var": sigma[2][2],
                                     "xdot_var": sigma[3][3], "ydot_var": sigma[4][4], "zdot_var": sigma[5][5],
                                     "kalman_gain": self.K, "t": t
                                     })
        else:
            self.estimated_x.append({"x_g": true_X[0][0], "y_g": true_X[1][0],
                                     "xdot_g": true_X[2][0], "ydot_g": true_X[3][0],
                                     "x": mu[0][0], "y": mu[1][0],
                                     "xdot": mu[2][0], "ydot": mu[3][0],
                                     "x_var": sigma[0][0], "y_var": sigma[1][1],
                                     "xdot_var": sigma[2][2], "ydot_var": sigma[3][3],
                                     "kalman_gain": self.K, "t": t
                                     })

        g = self.g(mu, u_t)
        G = self.G()
        mu_prediction = g
        sigma_prediction = np.add(np.matmul(np.matmul(G, sigma), G.T), self.R)
        d = self.h(mu_prediction)
        H = self.H(mu, d).T
        temp = np.add(np.matmul(np.matmul(H, sigma_prediction), H.T), self.Q)

        if temp.shape == (1, 1):
            temp = 1/temp
        else:
            temp = np.linalg.inv(temp)

        self.K = np.matmul(np.matmul(sigma_prediction, H.T), temp)

        mu_estimated = np.add(mu_prediction, np.matmul(self.K, np.subtract(z_t, d)))
        sigma_estimated = np.matmul(np.subtract(np.identity(self.n),
                                         np.matmul(self.K, H)), sigma_prediction)

        return mu_estimated, sigma_estimated


def run_football_poles(traj_precal=None, custom_state_noise= None, custom_observation_noise= None, title = "Given_Noise_With_BasePoles"):
    A = np.array([[1, 0, 0, 0.01, 0, 0],
                  [0, 1, 0, 0, 0.01, 0],
                  [0, 0, 1, 0, 0, 0.01],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])

    B = np.array([[0], [0], [0], [0], [0], [1]]).reshape((6, 1))

    if custom_state_noise is None:
        r = [0.01 ** 2, 0.01 ** 2, 0.01 ** 2, 0.1 ** 2, 0.1 ** 2, 0.1 ** 2]

    else:
        r = custom_state_noise
    custom_state_noise = r

    R = np.array([[r[0], 0, 0, 0, 0, 0],
                  [0, r[1], 0, 0, 0, 0],
                  [0, 0, r[2], 0, 0, 0],
                  [0, 0, 0, r[3], 0, 0],
                  [0, 0, 0, 0, r[4], 0],
                  [0, 0, 0, 0, 0, r[5]]])

    if custom_observation_noise is None:
        q = [0.1 ** 2, 0.1 ** 2, 0.1 ** 2, 0.1**2]
    else:
        q = custom_observation_noise

    custom_observation_noise = q

    Q = np.array([[q[0], 0, 0, 0],
                  [0, q[1], 0, 0],
                  [0, 0, q[2], 0],
                  [0, 0, 0, q[3]]])


    init_state = np.add(np.array([24.0, 4.0, 0.0, -16.04, 36.8, 8.61]),
                        np.random.normal(0, 10e-2, size=(6))).reshape((6, 1))

    var = np.array([[0.0001, 0, 0, 0, 0, 0],
                    [0, 0.0001, 0, 0, 0, 0],
                    [0, 0, 0.0001, 0, 0, 0],
                    [0, 0, 0, 0.0001, 0, 0],
                    [0, 0, 0, 0, 0.0001, 0],
                    [0, 0, 0, 0, 0, 0.0001]])

    if traj_precal is None:
        football = Football(init_state, var, None, True, custom_state_noise, custom_observation_noise)
    else:
        football = Football(traj_precal[0], var, None, True, custom_state_noise, custom_observation_noise, )

    Kfilter = ExtendedKalmanFilter(A, B, R, Q, 6, )

    i = 1
    for t in np.arange(0, 1.30, 0.01):
        ut = football.get_ut()
        zt = football.get_zt()
        mu, sigma = Kfilter.update(football.mu, football.sigma, ut, zt, t, football.true_X)
        football.mu = mu
        football.sigma = sigma
        if traj_precal is None:
            football.true_X = football.update_true()
        else:
            football.true_X = traj_precal[i]
            i += 1

    football.plot(Kfilter.estimated_x, title)
    return Kfilter


def plot_football_all(sensor_trajs):
    fig = go.Figure()
    field = [
        [32, -50, 0],
        [-32, -50, 0],
        [-32, 50, 0],
        [32, 50, 0],
        [32, -50, 0]
    ]
    goal = [
        [4, 50, 3],
        [-4, 50, 3],
        [-4, 50, 0],
        [4, 50, 0],
        [4, 50, 3]
    ]
    field_x, field_y, field_z = zip(*field)
    goal_x, goal_y, goal_z = zip(*goal)

    # drawing field
    fig.add_trace(go.Scatter3d(
        x=field_x, y=field_y, z=field_z,
        mode='lines',
        name='Playing Field',
        line=dict(color='red', width=2)
    ))
    # drawing goal post
    fig.add_trace(go.Scatter3d(
        x=goal_x, y=goal_y, z=goal_z,
        mode='lines',
        name='Goal Post',
        line=dict(color='blue', width=2)
    ))

    # drawing estimated trajectory
    names = ["Estimated with GPS", " Estimated with IMU", " Estimated with BasePoles"]

    for i, traj in enumerate(sensor_trajs):

        df = pd.DataFrame(traj.estimated_x)

        if i == 0:
            fig.add_trace(go.Scatter3d(
                x=df["x_g"],
                y=df["y_g"],
                z=df["z_g"],
                name='Ground',
                mode='lines',
            ))

        fig.add_trace(go.Scatter3d(
            x=df["x"],
            y=df["y"],
            z=df["z"],
            name=names[i],
            mode='lines',
        ))
        # drawing ground truth trajectory

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-60, 60], dtick=10),
            yaxis=dict(range=[-60, 60], dtick=10),
            zaxis=dict(range=[-60, 60], dtick=10)
        )
    )

    fig.update_layout(
        title='FootBall Trajectory',
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        )
    )

    fig.show()


def run_football_all(custom_state_noise=None, custom_observation_noise=None):
    np.random.seed(420)
    init_state = np.add(np.array([24.0, 4.0, 0.0, -16.04, 36.8, 8.61]),
                        np.random.normal(0, 10e-2, size=(6))).reshape((6, 1))
    var = np.array([[0.01, 0, 0, 0, 0, 0],
                    [0, 0.01, 0, 0, 0, 0],
                    [0, 0, 0.01, 0, 0, 0],
                    [0, 0, 0, 0.1, 0, 0],
                    [0, 0, 0, 0, 0.1, 0],
                    [0, 0, 0, 0, 0, 0.1]])
    if custom_state_noise is None:
        r = [0.01 ** 2, 0.01 ** 2, 0.01 ** 2, 0.1 ** 2, 0.1 ** 2, 0.1 ** 2]

    else:
        r = custom_state_noise
    custom_state_noise = r

    if custom_observation_noise is None:
        q = [0.01, 0.01, 0.01]
    else:
        q = custom_observation_noise

    custom_observation_noise = q
    football = Football(init_state, var, None,False, custom_state_noise, custom_observation_noise)
    traj = [football.mu]

    for t in np.arange(0, 1.30, 0.01):
        football.true_X = football.update_true()
        traj.append(football.true_X)

    Kfilter_imu = run_football_imu(traj)
    Kfilter_gps = run_football_gps(traj)
    Kfilter_poles = run_football_poles(traj)

    plot_football_all([Kfilter_gps, Kfilter_imu, Kfilter_poles])


def predict_goal(bel_mean, bel_variance):
    if bel_variance is None:
        raise NotImplementedError("not")
    else:
        raise NotImplementedError("not")


def plot_without_uncertainity(traj, goal, cols, title=''):
    fig = go.Figure()
    field = [
        [32, -50, 0],
        [-32, -50, 0],
        [-32, 50, 0],
        [32, 50, 0],
        [32, -50, 0]
    ]
    goal = [
        [4, 50, 3],
        [-4, 50, 3],
        [-4, 50, 0],
        [4, 50, 0],
        [4, 50, 3]
    ]
    field_x, field_y, field_z = zip(*field)
    goal_x, goal_y, goal_z = zip(*goal)

    # drawing field
    fig.add_trace(go.Scatter3d(
        x=field_x, y=field_y, z=field_z,
        mode='lines',
        name='Playing Field',
        line=dict(color='red', width=2)
    ))
    # drawing goal post
    fig.add_trace(go.Scatter3d(
        x=goal_x, y=goal_y, z=goal_z,
        mode='lines',
        name='Goal Post',
        line=dict(color='blue', width=2)
    ))

    for i in range(0, 1000):
        df = pd.DataFrame(traj[i][0], columns=cols)
        color = "red"
        if traj[i][1]:
            color = "green"
        fig.add_trace(go.Scatter3d(
            x=df["x"],
            y=df["y"],
            z=df["z"],
            showlegend=False,
            name='Ground',
            mode='lines',
            marker=dict(
                color=color,
                size=3
            ),
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-60, 60], dtick=10),
            yaxis=dict(range=[-60, 60], dtick=10),
            zaxis=dict(range=[-60, 60], dtick=10)
        )
    )
    fig.update_layout(title="Goals Scored Prediction " + title)
    fig.show()


def football_ground_traj_exp(custom_state_noise=None, custom_observation_noise=None):
    np.random.seed(404)
    init_state = np.add(np.array([24.0, 4.0, 0.0, -16.04, 36.8, 8.61]),
                        np.random.normal(0, 10e-2, size=(6))).reshape((6, 1))
    var = np.array([[0.01, 0, 0, 0, 0, 0],
                    [0, 0.01, 0, 0, 0, 0],
                    [0, 0, 0.01, 0, 0, 0],
                    [0, 0, 0, 0.1, 0, 0],
                    [0, 0, 0, 0, 0.1, 0],
                    [0, 0, 0, 0, 0, 0.1]])
    if custom_state_noise is None:
        r = [0.01 ** 2, 0.01 ** 2, 0.01 ** 2, 0.1 ** 2, 0.1 ** 2, 0.1 ** 2]

    else:
        r = custom_state_noise
    custom_state_noise = r

    if custom_observation_noise is None:
        q = [0.01, 0.01, 0.01]
    else:
        q = custom_observation_noise

    custom_observation_noise = q

    goal = 0
    traj = []
    for i in range(0, 1000):
        football = Football(init_state, var, None, False, custom_state_noise, custom_observation_noise)
        l_traj = [[football.true_X.flatten()], False]
        for t in np.arange(0, 1.30, 0.01):
            football.true_X = football.update_true()
            l_traj[0].append(football.true_X.flatten())

            if t > 1:
                position = football.true_X.flatten()
                goal_range_x = [-4, 4]
                goal_range_y = [49.5, 50.5]
                goal_range_z = [0, 3]

                if goal_range_x[0] < position[0] < goal_range_x[1]:
                    if goal_range_z[0] < position[2] < goal_range_z[1]:
                        if goal_range_y[0] < position[1] < goal_range_y[1]:
                            l_traj[1] = True
        traj.append(l_traj)
        if l_traj[1]:
            goal += 1
    plot_without_uncertainity(traj, goal, ["x", "y", "z", "xdot", "y_dot", "z_dot"], str(goal/10)+"%")


def football_gps_traj_exp(custom_state_noise= None, custom_observation_noise=None):
    np.random.seed(405)
    init_state = np.add(np.array([24.0, 4.0, 0.0, -16.04, 36.8, 8.61]),
                        np.random.normal(0, 10e-2, size=(6))).reshape((6, 1))
    var = np.array([[0.01, 0, 0, 0, 0, 0],
                    [0, 0.01, 0, 0, 0, 0],
                    [0, 0, 0.01, 0, 0, 0],
                    [0, 0, 0, 0.1, 0, 0],
                    [0, 0, 0, 0, 0.1, 0],
                    [0, 0, 0, 0, 0, 0.1]])
    if custom_state_noise is None:
        r = [0.01 ** 2, 0.01 ** 2, 0.01 ** 2, 0.1 ** 2, 0.1 ** 2, 0.1 ** 2]

    else:
        r = custom_state_noise
    custom_state_noise = r

    if custom_observation_noise is None:
        q = [0.01, 0.01, 0.01]
    else:
        q = custom_observation_noise

    custom_observation_noise = q

    C = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0]])

    goal = 0
    traj = []
    for i in range(0, 1000):
        football = Football(init_state, var, C, False, custom_state_noise, custom_observation_noise)

        l_traj = [[football.get_zt().flatten()], False]
        for t in np.arange(0, 1.30, 0.01):
            football.true_X = football.update_true()
            z_t = football.get_zt().flatten()
            l_traj[0].append(z_t)

            if t > 1:
                position = z_t
                goal_range_x = [-4, 4]
                goal_range_y = [49.5, 50.5]
                goal_range_z = [0, 3]

                if goal_range_x[0] < position[0] < goal_range_x[1]:
                    if goal_range_z[0] < position[2] < goal_range_z[1]:
                        if goal_range_y[0] < position[1] < goal_range_y[1]:
                            l_traj[1] = True
        traj.append(l_traj)
        if l_traj[1]:
            goal += 1
    plot_without_uncertainity(traj, goal, ["x", "y", "z"], str(goal/10)+"%")

def football_imu_traj_exp(custom_state_noise= None, custom_observation_noise=None):
    np.random.seed(405)
    init_state = np.add(np.array([24.0, 4.0, 0.0, -16.04, 36.8, 8.61]),
                        np.random.normal(0, 10e-2, size=(6))).reshape((6, 1))
    var = np.array([[0.01, 0, 0, 0, 0, 0],
                    [0, 0.01, 0, 0, 0, 0],
                    [0, 0, 0.01, 0, 0, 0],
                    [0, 0, 0, 0.1, 0, 0],
                    [0, 0, 0, 0, 0.1, 0],
                    [0, 0, 0, 0, 0, 0.1]])
    if custom_state_noise is None:
        r = [0.01 ** 2, 0.01 ** 2, 0.01 ** 2, 0.1 ** 2, 0.1 ** 2, 0.1 ** 2]

    else:
        r = custom_state_noise
    custom_state_noise = r

    if custom_observation_noise is None:
        q = [0.01, 0.01, 0.01]
    else:
        q = custom_observation_noise

    custom_observation_noise = q

    C = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0]])

    goal = 0
    traj = []
    for i in range(0, 1000):
        football = Football(init_state, var, C, False, custom_state_noise, custom_observation_noise)

        l_traj = [[football.get_zt().flatten()], False]
        for t in np.arange(0, 1.30, 0.01):
            football.true_X = football.update_true()
            z_t = football.get_zt().flatten()
            l_traj[0].append(z_t)

            if t > 1:
                position = z_t
                goal_range_x = [-4, 4]
                goal_range_y = [49.5, 50.5]
                goal_range_z = [0, 3]

                if goal_range_x[0] < position[0] < goal_range_x[1]:
                    if goal_range_z[0] < position[2] < goal_range_z[1]:
                        if goal_range_y[0] < position[1] < goal_range_y[1]:
                            l_traj[1] = True
        traj.append(l_traj)
        if l_traj[1]:
            goal += 1
    plot_without_uncertainity(traj, goal, ["x", "y", "z"], str(goal/10)+"%")

def predict_goal_with_uncertainty():
    np.random.seed(322)


    traj = []
    for i in range(0, 1000):

        kfilter = run_football_gps()
        estimate_x = kfilter.estimated_x
        for i in range(100, 130):
            position = [estimate_x[i]["x"], estimate_x[i]["y"], estimate_x[i]["z"]]

            goal_range_x = [-4, 4]
            goal_range_y = [50, 51]
            goal_range_z = [0, 3]

            goal_percentage = 0

            if goal_range_x[0] < position[0] < goal_range_x[1]:
                if goal_range_z[0] < position[2] < goal_range_z[1]:
                    if goal_range_y[0] < position[1] < goal_range_y[1]:
                        std_dev = [ np.sqrt(estimate_x[i]["x"]), np.sqrt(estimate_x[i]["y"]),
                                    np.sqrt(estimate_x[i]["z"])]

                        ecl_a = 2 * std_dev[0]
                        ecl_b = 2 * std_dev[1]
                        ecl_c = 2 * std_dev[2]

                        ecl_centre_x = position[0]
                        ecl_centre_y = position[1]
                        ecl_centre_z = position[2]

                        x_intervals = np.arange(0.5, 4, 1 / 10)
                        y_intervals = np.arange(50, 52, 1 / 10)
                        z_intervals = np.arange(0, 3, 1 / 10)
                        volume = (1 / 10) ** 3
                        max_volume = ((4 / 3) * np.pi * ecl_a * ecl_b * ecl_c)/2
                        total_volume = 0
                        for x in x_intervals:
                            for y in y_intervals:
                                for z in z_intervals:
                                    check = (((x - ecl_centre_x) ** 2) / ecl_a) + (((y - ecl_centre_y) ** 2) / ecl_b) + (
                                                ((z - ecl_centre_z) ** 2) / ecl_c)
                                    if check <= 1:
                                        total_volume += volume

                        goal_percentage = (total_volume/max_volume)*0.68



def extended_filter2d_exp(traj_precal=None, custom_state_noise= None, custom_observation_noise= None, field_mode = 0):
    A = np.array([[1, 0, 0.01, 0],
                  [0, 1, 0, 0.01],
                  [0, 0,  1, 0,],
                  [0, 0,  0, 1,],
                  ])

    B = np.array([[0], [0], [0], [0]]).reshape((4, 1))

    if custom_state_noise is None:
        r = [0.01 ** 2, 0.01 ** 2,0.1 ** 2, 0.1 ** 2]

    else:
        r = custom_state_noise
    custom_state_noise = r

    R = np.array([[r[0], 0, 0, 0, ],
                  [0, r[1], 0, 0, ],
                  [0, 0, r[2], 0, ],
                  [0, 0, 0, r[3],]])

    if custom_observation_noise is None:
        q = [0.1 ** 2, 0.1 ** 2,]
    else:
        q = custom_observation_noise

    custom_observation_noise = q

    Q = np.array([[q[0], 0,],
                  [0, q[1],],])


    init_state = np.add(np.array([0.0, -50.0, 0.0, 40.0]),
                        np.random.normal(0, 10e-2, size=(4))).reshape((4,1))

    var = np.array([[0.0001, 0, 0, 0, ],
                    [0, 0.0001, 0, 0,],
                    [0, 0, 0.0001, 0,],
                    [0, 0, 0, 0.0001,]])

    if traj_precal is None:
        football = Football(init_state, var, None, True, custom_state_noise, custom_observation_noise, field_2d=True, field_mode=field_mode)
    else:
        football = Football(traj_precal[0], var, None, True, custom_state_noise, custom_observation_noise, field_2d=True, field_mode=field_mode )

    Kfilter = ExtendedKalmanFilter(A, B, R, Q,4, field_mode)
    i = 1
    for t in np.arange(0, 1.30, 0.01):
        ut = football.get_ut()
        zt = football.get_zt()
        mu, sigma = Kfilter.update(football.mu, football.sigma, ut, zt, t, football.true_X)
        football.mu = mu
        football.sigma = sigma
        if traj_precal is None:
            football.true_X = football.update_true()
        else:
            football.true_X = traj_precal[i]
            i += 1

    plot_XY_projection(Kfilter.estimated_x, None)
    return Kfilter


def plot_XY_projection(data, sensor_traj, title="Given Noise"):
    df = pd.DataFrame(data)

    fig = go.Figure()
    field = [
        [32, -50],
        [-32, -50],
        [-32, 50],
        [32, 50],
        [32, -50]
    ]
    goal = [
        [4, 50],
        [-4, 50],
    ]
    field_x, field_y = zip(*field)
    goal_x, goal_y = zip(*goal)

    # drawing field
    fig.add_trace(go.Scatter(
        x=field_x, y=field_y,
        mode='lines',
        name='Playing Field',
        line=dict(color='red', width=2)
    ))
    # drawing goal post
    fig.add_trace(go.Scatter3d(
        x=goal_x, y=goal_y,
        mode='lines',
        name='Goal Post',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df["x_g"],
        y=df["y_g"],
        name='Ground',
        mode='lines',
    ))

    fig.add_trace(go.Scatter(
        x=df["x"],
        y=df["y"],
        name="Estimated Trajectory",
        mode='lines',
    ))
    # drawing ground truth trajectory
    if sensor_traj is not None:
        sen_df = pd.DataFrame(sensor_traj)
        fig.add_trace(go.Scatter(
            x=sen_df["x_s"],
            y=sen_df["y_s"],
            name="Sensor Trajectory",
            mode='lines',
        ))
    # sensor trajectory

    for i,row in df.iterrows():
        x_cen, y_cen = row["x"], row["y"]
        x_sigma, y_sigma = 2*np.sqrt(row["x_var"]), 2*np.sqrt(row["y_var"])
        angle = 30
        angle_rad = np.radians(angle)

        t = np.linspace(0, 2 * np.pi, 100)
        x_ellipse = x_cen + x_sigma * np.cos(t) * np.cos(angle_rad) - y_sigma * np.sin(t) * np.sin(angle_rad)
        y_ellipse = y_cen + x_sigma * np.cos(t) * np.sin(angle_rad) + y_sigma * np.sin(t) * np.cos(angle_rad)

        fig.add_trace(go.Scatter(x=x_ellipse, y=y_ellipse, mode='lines', showlegend=False, line=dict(color="rgba(0, 0, 150, 0.5)")))

        # fig.update_layout(
        #     scene=dict(
        #         xaxis=dict(range=[-32, 32], dtick=5),
        #         yaxis=dict(range=[-50, 50], dtick=5),
        #     )
        # )

        fig.update_layout(
            title='XY Projection of Football with Uncertainty Ellipses ' + title,
            scene=dict(
                xaxis_title='X Axis',
                yaxis_title='Y Axis',
            )
        )
        fig.update_layout(dragmode='pan')

    fig.show()


if __name__ == "__main__":
    # train_exp1 = run_train()
    # train_noise_more = run_train(None, None, [0.002, 0.35], [0.0002])
    # train_noise_less = run_train(None, None, [0.00005, 0.15], [0.000005])
    #train_failure_obs = run_train(t_start=1.5, t_end=2.5)

    # comp_plot_train([train_exp1[1].estimated_x, train_exp2_pos_noise_less[1].estimated_x,
    #                 train_exp2_pos_noise_more[1].estimated_x],
    #                 ["position", "position_mean"], ["given noise", "less noise", "more noise"],["time", "time"],
    #                 ["position","position"],["Actual Position", "Estimated Position"], [None,"position_var"])

    # train_exp3 = run_train(None, None, [0.1 ** 2, 0.7 ** 2], [0.01 ** 2])
    # train_exp4 = run_train(None, None, [0.1 ** 2, 0.5 ** 2], [0.04 ** 2])

    # run_football_all()          #Task 2(b)

    football_ground_traj_exp()      # Task 2(c)
    football_gps_traj_exp()         # Task 2(c)
    # predict_goal_with_uncertainty()
    # run_football_gps()
    # run_football_gps(custom_state_noise=[0.015**2, 0.015**2, 0.015**2, 0.15**2, 0.15**2, 0.015**2],
    #                  custom_observation_noise=[0.15**2, 0.15**2, 0.15**2], title="More_noise")
    # run_football_gps(custom_state_noise=[0.005 ** 2, 0.005 ** 2, 0.005 ** 2, 0.05 ** 2, 0.05 ** 2, 0.005 ** 2],
    #                  custom_observation_noise=[0.05 ** 2, 0.05 ** 2, 0.05 ** 2], title="Less_noise")

    #extended_filter2d_exp()
