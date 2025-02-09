import numpy as np
import numpy.random
import pandas
import plotly.graph_objects as go
import pandas as pd


class Train:
    def __init__(self, mu, sigma, C):
        self.mu = mu
        self.sigma = sigma
        self.C = C
        self.true_X = np.array([0, 0])

    def get_ut(self, t):
        noise_vel = np.random.normal(0, 0.5)
        if t < 0.25:
            return np.array([400]) + noise_vel
        elif 3 < t < 3.25:
            return np.array([-400]) + noise_vel
        else:
            return np.array([0]) + noise_vel

    def update_true(self, t):
        noise_pos = np.random.normal(0,0.1)
        noise_vel = np.random.normal(0,0.5)
        if t < 0.25:
            return self.true_X + np.array([self.true_X[1] * 0.01, 400]) + np.array([noise_pos,noise_vel])
        elif 3 < t < 3.25:
            return self.true_X + np.array([self.true_X[1] * 0.01, -400]) + np.array([noise_pos,noise_vel])
        else:
            return self.true_X + np.array([self.true_X[1] * 0.01, 0]) + np.array([noise_pos,noise_vel])

    def get_zt(self):
        return np.matmul(self.C, self.true_X.reshape((2, 1))) + np.random.normal(0,0.01)

    def plot(self, estimate_x):

        df = pd.DataFrame(estimate_x)

        fig1 = go.Figure()
        fig2 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df['t'],
            y=df['position_mean'],
            mode='lines',
            name='Estimated Position',
            error_y=dict(
                type='sqrt',
                array=df['position_var'],
                visible=True
            )

        ))

        fig1.add_trace(go.Scatter(
            x=df['t'],
            y=df['position'],
            mode='lines',
            name='True position',
        ))

        fig2.add_trace(go.Scatter(
            x=df['t'],
            y=df['velocity_mean'],
            mode='lines',
            name='Estimated_Velocity',
            error_y=dict(
                type='sqrt',
                array=df['velocity_var'],
                visible=True
            )

        ))

        fig1.update_layout(
            title='True and Estimated Position of Train',
            xaxis_title='Time',
            yaxis_title='Distance from A'
        )

        fig1.show()
        fig2.show()


def run_train():
    A = np.array([[1, 0.01], [0, 1]])
    B = np.array([[0], [1]])
    R = np.array([[0.01, 0], [0, 0.05]])
    Q = np.array([0.0001])
    C = np.array([[1 / 1500, 0]]).reshape((1,2))

    train = Train(np.random.normal(0,10e-4, 2).reshape((2,1)), np.array([[10e-4, 0], [0, 10e-4]]), C)
    Kfilter = KalmanFilter(A, B, R, Q, C,"train")

    for t in np.arange(0, 3.27, 0.01):
        ut = train.get_ut(t)
        zt = train.get_zt()
        mu, sigma = Kfilter.update(train.mu, train.sigma, ut, zt, t, train.true_X)
        train.mu = mu
        train.sigma = sigma
        train.true_X = train.update_true(t)

    train.plot(Kfilter.estimated_x)


class KalmanFilter:

    def __init__(self, A, B, R, Q, C, mode,n=2):

        self.A = A
        self.B = B
        self.K = 0
        self.Q = Q
        self.C = C
        self.R = R
        self.mode = mode
        self.n = n
        self.estimated_x = []

    def update(self, mu, sigma, u_t, z_t, t, true_X):

        if self.mode == "train":
            self.estimated_x.append({"position": true_X[0], "velocity": true_X[1],
                                     "position_mean":  mu[0][0], "velocity_mean": mu[1][0],"position_var": sigma[0][0],
                                     "velocity_var": sigma[1][1], "kalman_gain":self.K, "t": t})

        elif self.mode == "football":
            self.estimated_x.append({"x_g": true_X[0], "y_g": true_X[1], "z_g": true_X[2],
                                     "xdot_g": true_X[3], "ydot_g":true_X[4], "zdot_g":true_X[5],
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

        self.K = np.matmul(np.matmul(sigma_prediction, self.C.T), temp)
        if z_t is not None:
            mu_estimated = np.add(mu_prediction, np.matmul(self.K, np.subtract(z_t, np.matmul(self.C, mu_prediction))))
            sigma_estimated = np.matmul(np.subtract(np.identity(self.n),
                                             np.matmul(self.K, self.C)), sigma_prediction)

            return mu_estimated, sigma_estimated
        else:
            mu_estimated = mu_prediction
            sigma_estimated = np.matmul(np.subtract(np.identity(self.n),
                                                    np.matmul(self.K, self.C)), sigma_prediction)
            return mu_estimated, sigma_estimated


class Football:
    def __init__(self, mu, sigma, C, non_linear_obs=False,h=None):
        self.mu = mu
        self.sigma = sigma
        self.C = C
        self.true_X = np.array([24.0, 4.0, 0.0, -16.04, 36.8, 8.61])
        self.non_linear_obs = non_linear_obs
        if non_linear_obs:
            self.h = h

    def get_ut(self):
        return np.array([[-10*0.01]])

    def update_true(self):
        noise_pos = np.random.normal(0, 0.01, (3))
        noise_vel = np.random.normal(0, 0.1, (3))
        noise = np.concatenate((noise_pos, noise_vel), axis=0)
        xdot = self.true_X[3]
        ydot = self.true_X[4]
        zdot = self.true_X[5]
        return self.true_X + np.array([xdot*0.01, ydot*0.01, zdot*0.01, 0, 0, -10*0.01]) + noise

    def get_zt(self,arg):
        if not self.non_linear_obs:
            return np.matmul(self.C, self.true_X.reshape((6,1))) #+ np.random.normal(0, 0.1, (2))
        else:
            return self.h(arg)

    def plot(self, estimate_X):
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
        df = pandas.DataFrame(estimate_X)
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



        # for i in range(len(df)):
        #     fig.add_trace(go.Mesh3d(
        #         x= df["x_var"],
        #         y= df["y_var"],
        #         z= df["z_var"],
        #         alphahull=0,
        #         opacity=0.2,
        #         color='blue',
        #         showscale=False,
        #         name='Uncertainty Ellipse'
        #     ))


        fig.update_layout(
            title='Foot Trajectory with uncertainty',
            scene=dict(
                xaxis_title='X Axis',
                yaxis_title='Y Axis',
                zaxis_title='Z Axis'
            )
        )

        fig.show()


class ExtendedKalmanFilter:
    def __init__(self):
        pass


def run_football_gps():
    A = np.array([[1, 0, 0, 0.01, 0, 0],
                  [0, 1, 0, 0, 0.01, 0],
                  [0, 0, 1, 0, 0, 0.01],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])

    B = np.array([[0], [0], [0], [0], [0], [1]]).reshape((6, 1))

    R = np.array([[0.01, 0, 0, 0, 0, 0],
                  [0, 0.01, 0, 0, 0, 0],
                  [0, 0, 0.01, 0, 0, 0],
                  [0, 0, 0, 0.1, 0, 0],
                  [0, 0, 0, 0, 0.1, 0],
                  [0, 0, 0, 0, 0, 0.1]])

    Q = np.array([[0.1, 0, 0],
                  [0, 0.1, 0],
                  [0, 0, 0.1]])

    C = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0]])

    init_state = np.add(np.array([24.0, 4.0, 0.0, -16.04, 36.8, 8.61]),numpy.random.normal(0, 10e-4, size=(6))).reshape((6,1))
    var = np.array([[0.01, 0, 0, 0, 0, 0],
                  [0, 0.01, 0, 0, 0, 0],
                  [0, 0, 0.01, 0, 0, 0],
                  [0, 0, 0, 0.1, 0, 0],
                  [0, 0, 0, 0, 0.1, 0],
                  [0, 0, 0, 0, 0, 0.1]])
    football = Football(init_state, var, C)
    Kfilter = KalmanFilter(A, B, R, Q, C, "football", 6)

    for t in np.arange(0, 1.31, 0.01):
        ut = football.get_ut()
        zt = football.get_zt()
        mu, sigma = Kfilter.update(football.mu, football.sigma, ut, zt, t, football.true_X)
        football.mu = mu
        football.sigma = sigma
        football.true_X = football.update_true()

    football.plot(Kfilter.estimated_x)


def run_football_imu():
    A = np.array([[1, 0, 0, 0.01, 0, 0],
                  [0, 1, 0, 0, 0.01, 0],
                  [0, 0, 1, 0, 0, 0.01],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])

    B = np.array([[0], [0], [0], [0], [0], [1]])

    R = np.array([[0.01, 0, 0, 0, 0, 0],
                  [0, 0.01, 0, 0, 0, 0],
                  [0, 0, 0.01, 0, 0, 0],
                  [0, 0, 0, 0.1, 0, 0],
                  [0, 0, 0, 0, 0.1, 0],
                  [0, 0, 0, 0, 0, 0.1]])

    Q = np.array([[0.1, 0, 0, 0, 0, 0],
                  [0, 0.1, 0, 0, 0, 0],
                  [0, 0, 0.1, 0, 0, 0],
                  [0, 0, 0, 0.1, 0, 0],
                  [0, 0, 0, 0, 0.1, 0],
                  [0, 0, 0, 0, 0, 0.1]])

    C = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])

    init_state = np.add(np.array([24.0, 4.0, 0.0, -16.04, 36.8, 8.61]),
                        numpy.random.normal(0, 10e-4, size=(6))).reshape((6, 1))
    var = np.array([[0.01, 0, 0, 0, 0, 0],
                    [0, 0.01, 0, 0, 0, 0],
                    [0, 0, 0.01, 0, 0, 0],
                    [0, 0, 0, 0.1, 0, 0],
                    [0, 0, 0, 0, 0.1, 0],
                    [0, 0, 0, 0, 0, 0.1]])

    football = Football(init_state, var, C)
    Kfilter = KalmanFilter(A, B, R, Q, C, "football", 6)

    for t in np.arange(0, 1.31, 0.01):
        ut = football.get_ut()
        zt = football.get_zt()
        mu, sigma = Kfilter.update(football.mu, football.sigma, ut, zt, t, football.true_X)
        football.mu = mu
        football.sigma = sigma
        football.true_X = football.update_true()

    football.plot(Kfilter.estimated_x)

def h_base_station(d):
    bs1 = []

def run_football_poles():
    A = np.array([[1, 0, 0, 0.01, 0, 0],
                  [0, 1, 0, 0, 0.01, 0],
                  [0, 0, 1, 0, 0, 0.01],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])

    B = np.array([[0], [0], [0], [0], [0], [1]])

    R = np.array([[0.01, 0, 0, 0, 0, 0],
                  [0, 0.01, 0, 0, 0, 0],
                  [0, 0, 0.01, 0, 0, 0],
                  [0, 0, 0, 0.1, 0, 0],
                  [0, 0, 0, 0, 0.1, 0],
                  [0, 0, 0, 0, 0, 0.1]])

    Q = np.array([[0.1, 0, 0, 0],
                  [0, 0.1, 0, 0],
                  [0, 0, 0.1, 0],
                  [0, 0, 0, 0.1]])
    #this is non linear


def predict_goal(bel_mean, bel_variance):
    if bel_variance is None:
        raise NotImplementedError("not")
    else:
        raise NotImplementedError("not")

if __name__ == "__main__":
    run_train()
    #run_football_gps()
    #run_football_imu()