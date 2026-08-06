# [訊息] X.shape = (225, 300), y.shape = (225,)
# [Debug] i_crit = 2, RMSEP_scree length = 20
# [結果] 選取的波段索引: [173  60]
# [選取的波長] [748.327809 516.914361]
# [結果] 選取的波段索引: [173  60]
# [訊息] 波長 shape: (300,)
# [選取的波段] 索引: [173  60]
# [選取的波段] 波長: [748.327809 516.914361]
#


# [訊息] X.shape = (280, 300), y.shape = (280,)
# [Debug] i_crit = 2, RMSEP_scree length = 20
# [結果] 選取的波段索引: [23 23]
# [選取的波長] [451.173663 451.173663]
# [結果] 選取的波段索引: [23 23]
# [訊息] 波長 shape: (294,)
# [選取的波段] 索引: [23 23]
# [選取的波段] 波長: [451.173663 451.173663]



# 原理
# 連續投影演算法（successiveprojectionsalgorithm, SPA）是前向特徵變數選擇方法。
# 利用SPA的投影分析，透過將投影投影到其他波長上，比較投影投影的大小，以投影投影最大的波長為待選擇的最終波長，然後根據校正模型的特徵波長進行選擇。 SPA的選擇是含有最少箭頭資訊及最小共線性的變數組合。
# 演算法簡單步驟如下：記初始迭代函數為xk(0)，需要擷取的參數個數為N，光譜矩陣為J列。
# 任選頻譜矩陣的1列（第j列），把建模集的第j列賦值給xj，記為xk(0)。
# 將未選入的列管理位置的集合記為s, s={j,1≤j≤J,j∉{k(0),⋯,k(n−1)}}分別計算xj對剩餘列管理的投影：Pxj=xj−(xTjxk(n−1))xk(n−1)(xTk(n−1)(xTk(n−1)(11)1)11)(11)1)15)15)125)，提取光譜的最大光譜。 k(n)=arg(max(|P(xj)|),j∈s) 令xj=px,j∈s。
# n=n+1，如果n<N，則以公式（1）循環計算。
# 最後，擷取的指標為{xk(n)=0,⋯,N−1}。對應每次循環中的k(0)和N，分別建立多元線性迴歸分析（MLR）模型，得到建模集交互驗證均方根（RMSECV），對應不同的子選項集，其中最小的 RMSECV 值對應的 k(0) 和 N 就是最優值。一般 SPA 選擇的特徵波長分數 N 不能很大。 ------------------------摘自《光譜及成像技術在農業上的應用》P130

# 來源: https://github.com/mepleleo/SPA/blob/main/SPA.py


import scipy.stats
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import scipy.stats
from scipy.linalg import qr
from progress.bar import Bar

# SPA原為matlab內工具的功能
# 此為github上別人還原出來的

# 設定字型
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16

class SPA:
    def _projections_qr(self, X, k, M):
        '''
        原版连续投影算法使用MATLAB内置的QR函数
        该版本改用scipy.linalg.qr函数
            https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.linalg.qr.html
        X : 预测变量矩阵
        K ：投影操作的初始列的索引
        M : 结果包含的变量个数
        return ：由投影操作生成的变量集的索引
        '''
        X_projected = X.copy()
        # 计算列向量的平方和
        norms = np.sum((X ** 2), axis=0)
        # 找到norms中数值最大列的平方和
        norm_max = np.amax(norms)
        # 缩放第K列 使其成为“最大的”列
        X_projected[:, k] = X_projected[:, k] * 2 * norm_max / norms[k]
        # 矩阵分割 ，order 为列交换索引
        _, __, order = qr(X_projected, 0, pivoting=True)
        return order[:M].T

    def _validation(self, Xcal, ycal, var_sel, Xval=None, yval=None):
        '''
        [yhat,e] = validation(Xcal,var_sel,ycal,Xval,yval) -->  使用单独的验证集进行验证
        [yhat,e] = validation(Xcal,ycalvar_sel) --> 交叉验证
        '''
        N = Xcal.shape[0]  # N 测试集的个数
        if Xval is None:  # 判断是否使用验证集
            NV = 0
        else:
            NV = Xval.shape[0]  # NV 验证集的个数

        yhat = e = None
        # 使用单独的验证集进行验证
        if NV > 0:
            Xcal_ones = np.hstack(
                [np.ones((N, 1)), Xcal[:, var_sel].reshape(N, -1)])
            # 对偏移量进行多元线性回归
            b = np.linalg.lstsq(Xcal_ones, ycal, rcond=None)[0]
            # 对验证集进行预测
            np_ones = np.ones((NV, 1))
            Xval_ = Xval[:, var_sel]
            X = np.hstack([np.ones((NV, 1)), Xval[:, var_sel]])
            yhat = X.dot(b)
            # 计算误差
            e = yval - yhat
        else:
            # 为yhat 设置适当大小
            yhat = np.zeros((N, 1))
            for i in range(N):
                # 从测试集中 去除掉第 i 项
                cal = np.hstack([np.arange(i), np.arange(i + 1, N)])
                X = Xcal[np.ix_(cal, var_sel.astype(int))]
                y = ycal[cal]
                xtest = Xcal[i, var_sel]
                # ytest = ycal[i]
                X_ones = np.hstack([np.ones((N - 1, 1)), X.reshape(N - 1, -1)])
                # 对偏移量进行多元线性回归
                b = np.linalg.lstsq(X_ones, y, rcond=None)[0]
                # 对验证集进行预测
                yhat[i] = np.hstack([np.ones(1), xtest]).dot(b)
            # 计算误差
            e = ycal - yhat

        return yhat, e

    def spa(self, Xcal, ycal, m_min=1, m_max=None, Xval=None, yval=None, autoscaling=1):
        '''
        [var_sel,var_sel_phase2] = spa(Xcal,ycal,m_min,m_max,Xval,yval,autoscaling) --> 使用单独的验证集进行验证
        [var_sel,var_sel_phase2] = spa(Xcal,ycal,m_min,m_max,autoscaling) --> 交叉验证

        如果 m_min 为空时， 默认 m_min = 1
        如果 m_max 为空时：
            1. 当使用单独的验证集进行验证时， m_max = min(N-1, K)
            2. 当使用交叉验证时，m_max = min(N-2, K)
        autoscaling : 是否使用自动刻度 yes = 1，no = 0, 默认为 1
        '''
        assert (autoscaling == 0 or autoscaling == 1), "请选择是否使用自动计算"
        N, K = Xcal.shape
        if m_max is None:
            if Xval is None:
                m_max = min(N - 1, K)
            else:
                m_max = min(N - 2, K)
        assert (m_max < min(N - 1, K)), "m_max 参数异常"

        # 第一步： 对测试集进行投影操作
        # 在均值中心化 和 自动窗口 之后 对 Xcal的列进行投影操作
        normalization_factor = None
        if autoscaling == 1:
            normalization_factor = np.std(
                Xcal, ddof=1, axis=0).reshape(1, -1)[0]
        else:
            normalization_factor = np.ones((1, K))[0]

        Xcaln = np.empty((N, K))
        for k in range(K):
            x = Xcal[:, k]
            Xcaln[:, k] = (x - np.mean(x)) / normalization_factor[k]

        SEL = np.zeros((m_max, K))
        # 进度条
        with Bar('Projections :', max=K) as bar:
            for k in range(K):
                SEL[:, k] = self._projections_qr(Xcaln, k, m_max)
                bar.next()

        # 第二步： 进行评估
        PRESS = float('inf') * np.ones((m_max + 1, K))
        with Bar('Evaluation of variable subsets :', max=(K) * (m_max - m_min + 1)) as bar:
            for k in range(K):
                for m in range(m_min, m_max + 1):
                    var_sel = SEL[:m, k].astype(int)
                    _, e = self._validation(Xcal, ycal, var_sel, Xval, yval)
                    PRESS[m, k] = np.sum(e ** 2)
                    bar.next()
        PRESSmin = np.min(PRESS, axis=0)
        m_sel = np.argmin(PRESS, axis=0)
        k_sel = np.argmin(PRESSmin)
        # 第 k_sel 波段为初始波段时最佳，波段数目为 m_sel(k_sel)
        var_sel_phase2 = SEL[:m_sel[k_sel], k_sel].astype(int)

        # 最后消去变量
        # 第 3.1 步 计算相关指数
        Xcal2 = np.hstack([np.ones((N, 1)), Xcal[:, var_sel_phase2]])
        b = np.linalg.lstsq(Xcal2, ycal, rcond=None)[0]
        std_deviation = np.std(Xcal2, ddof=1, axis=0)

        relev = np.abs(b * std_deviation.T)
        relev = relev[1:]

        index_increasing_relev = np.argsort(relev, axis=0)
        index_decreasing_relev = index_increasing_relev[::-1].reshape(1, -1)[0]

        PRESS_scree = np.empty(len(var_sel_phase2))
        yhat = e = None
        for i in range(len(var_sel_phase2)):
            var_sel = var_sel_phase2[index_decreasing_relev[:i + 1]]
            _, e = self._validation(Xcal, ycal, var_sel, Xval, yval)

            PRESS_scree[i] = np.sum(e ** 2)

        RMSEP_scree = np.sqrt(PRESS_scree / len(e))

        # 第 3.3： F-test 验证
        PRESS_scree_min = np.min(PRESS_scree)
        alpha = 0.25
        dof = len(e)
        fcrit = scipy.stats.f.ppf(1 - alpha, dof, dof)
        PRESS_crit = PRESS_scree_min * fcrit

        # 找到不明显比 PRESS_scree_min 大的最小变量
        valid_indices = np.nonzero(PRESS_scree < PRESS_crit)[0]
        if len(valid_indices) == 0:
            i_crit = 0  # fallback 到第一個
        else:
            i_crit = np.min(valid_indices)
        i_crit = np.min(valid_indices) if len(valid_indices) > 0 else 0
        # 重要：防止 i_crit 超過 RMSEP_scree 長度
        i_crit = min(max(m_min, i_crit), len(RMSEP_scree) - 1)

        print(f"[Debug] i_crit = {i_crit}, RMSEP_scree length = {len(RMSEP_scree)}")

        if i_crit == 0:
            var_sel = var_sel_phase2[index_decreasing_relev[:1]]  # 至少選1個
        else:
            var_sel = var_sel_phase2[index_decreasing_relev[:i_crit]]

        # plt.title('SPA')
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        fig1 = plt.figure()
        plt.xlabel('Number of variables included in the model')
        plt.ylabel('RMSE')
        plt.title('Final number of selected variables:{}(RMSE={})'.format(len(var_sel), RMSEP_scree[i_crit]))
        plt.plot(RMSEP_scree)
        plt.scatter(i_crit, RMSEP_scree[i_crit], marker='s', color='r')
        plt.grid(True)

        fig2 = plt.figure()
        plt.plot(Xcal[0, :])
        plt.scatter(var_sel, Xcal[0, var_sel], marker='s', color='r')
        plt.legend(['First calibration object', 'Selected variables'])
        plt.xlabel('Variable index')
        plt.grid(True)
        plt.title('SPA')
        plt.savefig('SPA_result.png', dpi=300)
        plt.show()
        return var_sel, var_sel_phase2

    def __repr__(self):
        return "SPA()"


'''
（1）变量Xcal，Ycal指的是（Xcalibration和Ycalibration）, Xval个Yval指的是（Xvalidation和Yvalidation）,
    从变量的命名可以看出，Xcal和Ycal是需要计算的光谱矩阵（训练集），Xval和Yval是验证的光谱矩阵（测试集）。
（2）Xcal（训练集矩阵）和Xval（测试集矩阵）都是M*N的光谱矩阵（M为样本数，N为维度（波段））。
（3）Ycal和Yval在程序里没有注释，不是很清楚含义。Ycal和Yval都是M*1的维的矩阵，应该是是训练集和测试集的训练标签。
（4）分析前光谱数据需要平滑去噪，否则误差较大。
'''
# if __name__ == "__main__":
#     data = pd.read_excel(r"F:\实验\data\all.xlsx")
#     x = data.drop(['names', 'labels'], axis=1)
#     y = data.loc[:, 'labels']
#
#     absorbances = x.columns.values
#
#     from sklearn.model_selection import train_test_split
#     from sklearn.preprocessing import MinMaxScaler
#
#     Xcal, Xval, ycal, yval = train_test_split(x, y, test_size=0.4, random_state=0)
#
#     min_max_scaler = MinMaxScaler(feature_range=(-1, 1))  # 这里feature_range根据需要自行设置，默认（0,1）
#
#     Xcal = min_max_scaler.fit_transform(Xcal)
#     Xval = min_max_scaler.transform(Xval)
#
#     # var_sel, var_sel_phase2 = SPA().spa(
#     #     Xcal, ycal, m_min=2, m_max=50, Xval=Xval, yval=yval, autoscaling=1)
#     var_sel, var_sel_phase2 = SPA().spa(
#         Xcal, ycal, m_min=2, m_max=50, Xval=Xval, yval=yval, autoscaling=1)
#     print(absorbances[var_sel])

# ===============加入內容=========================
# === 主程式 ===
if __name__ == "__main__":
    # === 1. 路徑設定 ===
    healthy_dir = Path(r"D:/Users/Amanda/PycharmProjects/test/test_0415_LDA/重製1003/ks_npy_train_subset/healthy")
    unhealthy_dir = Path(r"D:/Users/Amanda/PycharmProjects/test/test_0415_LDA/重製1003/ks_npy_train_subset/unhealthy")


    # === 2. 讀取全部 healthy / unhealthy 光譜 ===
    def load_all_spectra(directory):
        spectra = []
        for file in directory.glob("*.npy"):
            spectrum = np.load(file)
            spectra.append(spectrum)
        return np.array(spectra)

    X_healthy = load_all_spectra(healthy_dir)
    X_unhealthy = load_all_spectra(unhealthy_dir)

    # === 3. 合併資料 ===
    X = np.vstack([X_healthy, X_unhealthy])
    y_healthy = np.zeros(len(X_healthy))
    y_unhealthy = np.ones(len(X_unhealthy))
    y = np.hstack([y_healthy, y_unhealthy])

    print(f"[訊息] X.shape = {X.shape}, y.shape = {y.shape}")

    # === 4. 建模集測試集分割 → 再歸一化 ===
    Xcal, Xval, ycal, yval = train_test_split(X, y, test_size=0.4, random_state=0)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    Xcal = scaler.fit_transform(Xcal)
    Xval = scaler.transform(Xval)

    ycal = ycal.reshape(-1, 1)
    yval = yval.reshape(-1, 1)

    # === 5. 執行 SPA ===
    spa_instance = SPA()
    selected_bands, selected_bands_phase2 = spa_instance.spa(Xcal=Xcal, ycal=ycal, m_min=2, m_max=20, Xval=Xval, yval=yval, autoscaling=1)

    print(f"[結果] 選取的波段索引: {selected_bands}")

    # === 6. 顯示對應波長 ===
    wavelengths = np.load(r"D:/Users/Amanda/PycharmProjects/test/test_0415_LDA/wavelengths.npy")
    selected_wavelengths = wavelengths[selected_bands]

    print(f"[選取的波長] {selected_wavelengths}")

    # === 5. 執行 SPA 完畢後，檢查選取的波段索引 ===
    print(f"[結果] 選取的波段索引: {selected_bands}")  # 應印出數字，例如 [12 25]

    # === 6. 顯示對應波長 ===
    wavelengths = np.load(r"D:/Users/Amanda/PycharmProjects/test/test_0415_LDA/wavelengths.npy")
    print(f"[訊息] 波長 shape: {wavelengths.shape}")

    # === 確認 selected_bands 是否為空或 None ===
    if selected_bands is not None and len(selected_bands) > 0:
        selected_wavelengths = wavelengths[selected_bands]
        print(f"[選取的波段] 索引: {selected_bands}")
        print(f"[選取的波段] 波長: {selected_wavelengths}")
    else:
        print("[警告] 沒有選出波段！請檢查 SPA 結果。")







