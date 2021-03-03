import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error


def MZ(t, M0, T1):
    """
    Evolution of longitudinal magnetization in a IR sequence
    """
    return M0 * (1.0 - 2 * np.exp(-t / T1))


def MT(FA, M0, T1):
    """
    Signal equation of spoiled gradient echo sequence
    """
    return (
        M0
        * np.sin(FA * np.pi / 180)
        * (1.0 - np.exp(-20.0 / T1))
        / (1.0 - (np.exp(-20.0 / T1) * np.cos(FA * np.pi / 180)))
    )


def M_xy(t, M0, T2):
    """
    FID signal equation
    """
    return M0 * np.exp(-t / T2)


def Ernst_T1(TR, alpha_e):
    """
    Ernst angle formula solved for T1
    """
    return -TR / np.log(np.cos(alpha_e))


def SNR(X, k, n):
    """
    Equation used to curve fit SNR
    """
    signal, NSA = X  # signal = mean signal intensity / SD
    return k * signal * (NSA ** n)


def IR():
    """
    Signal intensities for all 5 compartments for 13 different pictures
    with 13 different TI-times
    """
    s = np.array(
        [2.40774137,2.287696084,2.203613927,2.048710132,1.899829585,1.591776247,
        2.021218754,2.572949552,3.298381484,3.635993426,3.788266224,3.8307278,3.834208811]
    )

    TI = np.array([50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000])

    comp1 = s * np.array([-159.1,-134.2,-109.1,-64.7,25.0,40.1,88.6,126.8,187.6,219.4,245.4,253.6,256.1])
    comp2 = s * np.array([-368.3,-356.9,-343.8,-318.1,-292.0,-242.5,-199.3,-158.4,-68.8,14.2,131.9,219.5,333.5])
    comp3 = s * np.array([-77.5,-51.9,-29.8,9.9,40.2,85.7,115.4,135.1,160.1,167.6,172.3,171.7,171.8])
    comp4 = s * np.array([-265.0,-240.6,-216.7,-170.5,-128.2,-53.5,9.6,62.3,159.7,223.8,296.5,328.3,346.7])
    comp5 = s * np.array([-346.5,-328.9,-312.1,-278.5,-244.4,-182.3,-128.0,-80.0,30.8,109.3,225.1,299.5,372.2])

    comp = [comp1, comp2, comp3, comp4, comp5]
    MSE = []
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    x_new = np.linspace(0, 3000, 10000)
    for i, j, k in zip(comp, colors, np.arange(1, 6)):
        plt.scatter(TI, i, c=j)
        # popt, _ = curve_fit(MZ, TI, i, p0=np.array([200, 220, 300]))
        popt, _ = curve_fit(MZ, TI, i, p0=np.array([300, 220]))
        # M_z0, T1, M0 = popt
        M0, T1 = popt
        y_new = MZ(x_new, *popt)
        plt.plot(x_new, y_new, "--", c=j, label=f"Fit Comp. {k:d} : $T_1$={T1:3.2f}")
        MSE.append(mean_squared_error(i,y_new[TI]))
    print(MSE)
    print(np.mean(MSE))
    plt.grid()
    plt.legend(loc="best")
    plt.xlabel("TI")
    plt.ylabel(r"Singal Intensity $M_z$")
    plt.show()


def T1_GRE():
    FA = np.array([1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90])
    comp1 = np.array([4.4, 22.8, 41.0, 52.6, 58, 58.7, 53.8, 48.1, 43.3, 38.3, 33.4, 30.6])
    comp2 = np.array([3.7, 17.4, 25.5, 27.3, 26.5, 23.6, 19.7, 16.9, 14.5, 11.7, 9.1, 7.3])
    comp3 = np.array([4.4, 23.1, 43.2, 58.4, 68.2, 75.4, 73.9, 69.9, 64.5, 58.8, 53.2, 48.8])
    comp4 = np.array([5.0, 26.0, 44.0, 53.2, 55.9, 52.2, 46.2, 40.5, 35.3, 31.2, 27.8, 25.0])
    comp5 = np.array([4.3, 20.5, 32.1, 35.2, 34.5, 30.5, 25.7, 21.1, 17.3, 13.7, 11.5, 8.6])

    comp = [comp1, comp2, comp3, comp4, comp5]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    MSE = []

    x_new = np.linspace(1, 90, 10000)

    for i, j, k in zip(comp, colors, range(1, 6)):
        popt, _ = curve_fit(MT, FA, i, p0=np.array([200, 300]))
        M0, T1 = popt
        y_new = MT(x_new, M0, T1)
        plt.scatter(FA, i)
        plt.plot(x_new, y_new, "--", c=j, label=f"Fit Comp. {k:d} : $T_1$={T1:.2f}")
        MSE.append(mean_squared_error(i,y_new[FA]))
    print(MSE)
    plt.grid()
    plt.legend(loc="best")
    plt.xlabel("Flip angle (degrees)")
    plt.ylabel(r"Singal Intensity")
    plt.show()

    T = []
    for i in comp:
        T.append(Ernst_T1(20.0, FA[np.argmax(i)] * np.pi / 180))

    print(T)  # Estimated T1 for each compartment using estimated Ernst angle


def T1(TR, FA_1, FA_2, M1, M2):
    """
    Takes ratio of two FA's (FA_1 and FA_2) and calculates T1.

    M1: Signal intensity of FA_1
    M2: Signal intensity of FA_2
    """
    a = M1 * np.sin(FA_2) / (M2 * np.sin(FA_1))
    return -TR / np.log((a - 1.0) / ((a * np.cos(FA_1)) - np.cos(FA_2)))


def calc_T1_from_FA_ratio():
    """
    Function takes two FAs and corresponding signal intensities
    and then computes estimated T1 relaxation for each compartment
    using ratio of two signal intensities.
    """
    FA = np.array([5, 20]) * np.pi / 180
    comp1 = np.array([22.8, 58.0])
    comp2 = np.array([17.4, 26.5])
    comp3 = np.array([23.1, 68.2])
    comp4 = np.array([26.0, 55.9])
    comp5 = np.array([20.5, 34.5])
    comp = [comp1, comp2, comp3, comp4, comp5]
    T = []
    for i in comp:
        T.append(T1(20.0, FA[0], FA[1], i[0], i[1]))
    return T

def calc_T1_set_MZ_to_zero():
    """
    Calculates T1 times for each compartment using
    time point where signal intensity in IR sequence is zero.
    """
    T1 = np.array([200., 907.15, 128.14, 362.98, 647.01]) # Estimated by eye
    return T1/np.log(2)

def SNR_vs_NSA():
    """
    Function plots SNR vs. NSA. SNR is obtained from finding
    variance of pixel intensity for each MRI.
    """
    NSA = np.array([1, 2, 3, 4])
    noise = np.array([0.7, 0.5, 0.4, 0.3])  # **2
    comp1 = np.array([39.3, 40.8, 41.3, 41.2])
    comp2 = np.array([55.3, 56.5, 56.8, 56.4])
    comp3 = np.array([69.4, 69.2, 69.1, 68.7])
    comp4 = np.array([53.5, 53.3, 53.7, 53.0])
    comp5 = np.array([65.1, 65.4, 65.7, 65.7])

    comp = [comp1, comp2, comp3, comp4, comp5]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, j, f in zip(comp, range(1, 6), colors):
        popt, _ = curve_fit(SNR, (NSA, i / noise), i / noise, p0=np.array([2, 0.5]))
        k, n = popt
        y_new = SNR((NSA, i / noise), k, n)
        plt.plot(NSA, y_new, c=f)
        plt.plot(NSA, i / noise, "o", label="Compartment %i, n=%.2f" % (j, n))

    plt.grid()
    plt.legend()
    plt.ylabel("SNR")
    plt.xlabel("NSA")
    plt.show()

    # curve fit to see if SNR goes as root of NSA


def EPI():
    """
    Function computes plots signal evolution for four different areas in MR images
    obtained by GE-EPI sequcene for 10 different TE-times.
    Using FID signal equation for curve Fit and estimate T2-times.
    """
    TE = np.array([4.22, 33.81, 63.39, 92.98, 122.6, 152.2, 181.7, 211.3, 240.9, 270.5])
    upper_left = np.array([697.3, 367.0, 217.5, 115.8, 51.8, 23.2, 14.8, 8.7, 6.1, 4.6])
    center = np.array([1110.2, 907.8, 813.6, 745.2, 692.8, 637.0, 564.9, 521.0, 450.2, 401.6])
    lower_right = np.array([723.0, 419.2, 224.1, 126.4, 61.8, 32.4, 15.1, 8.8, 3.9, 3.8])
    upper_center = np.array([782.2, 499.4, 279.5, 154.5, 88.6, 58.2, 43.8, 38.2, 38.2, 36.0])

    area = [upper_left, center, upper_center, lower_right]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    name = ["Upper left area", "Center area", "Up center area", "Lower right area"]
    x_new = np.linspace(4.22, 270.5, 10000)
    for i, j, k in zip(area, colors, name):
        popt, _ = curve_fit(M_xy, TE, i, p0=np.array([200, 300]))
        M0, T2 = popt[0], popt[1]
        y_new = M_xy(x_new, M0, T2)
        plt.scatter(TE, i)
        plt.plot(x_new, y_new, "--", c=j, label="Fit: %s" % k + f", $T_2$={T2:.2f}")
    plt.legend(loc="best")
    plt.grid()
    plt.ylabel("Mean Signal Intensity")
    plt.xlabel("TE [ms]")
    plt.show()


def T1_contrast():
    """
    Function estimates inversion times which will give maximum contrast between
    tissue with 900 ms relaxation and another tissue with 1200 ms relaxation.
    """
    t = np.linspace(0, 10000, 10000)
    s1 = MZ(t, 200, 900)
    s2 = MZ(t, 200, 1200)
    r = np.argmax(s1 - s2)
    print(f"{r:.2f}")
    plt.plot(t, s1, label="White matter")
    plt.plot(t, s2, label="Grey matter")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    IR()
    T1_GRE()
    #T1 = calc_T1_from_FA_ratio()
    #T1 = calc_T1_set_MZ_to_zero()
    #print(T1)
    #SNR_vs_NSA()
    #EPI()
    #T1_contrast()
