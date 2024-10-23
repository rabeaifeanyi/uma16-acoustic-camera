import numpy as np
import acoular as ac
from tapy.devices.sensors import OHT20 #type: ignore
import os
import datetime

oht20_sensor = OHT20(port="/dev/ttyACM0")
ac.config.global_caching = "none" # type: ignore

def mole_fraction_of_water_vapor(h, T, p=101325):
    """mole fraction of water vapor in the air for real(!) gases according to Giacomo et al. (1982)

    Equations A1-3 from appendix in Cramer (1993).

    The equations are only valid over the temperature range0 øCto 30 øC (273.15 K 
    to 303.15 K) and for the pressurerange 60 000 to 110000 Pa.

    :param h: relative humidity as fraction [0,1]
    :param T: thermodynamic temperature in K
    :param p: atmospheric pressure in Pa (default is the standard pressure 101325 Pa)
    :return: mole fraction of water vapor
    """
    if T < 273.15 or T > 303.15:
        raise ValueError('Temperature out of range')
    if p < 60000 or p > 110000:
        raise ValueError('Pressure out of range')

    f = 1.00062 + 3.14*10**(-8)*p + 5.6*10**(-7)*T**2 # enhancement factor
    #saturated vapor pressure from F. Giacomo et al. (1982)
    #p_sv = np.exp(1.2811805*10**(-5)*T**2 - 1.9509874*10**(-2)*T + 34.04926034 - 6.3536311*10**3/T) 
    
    # updated saturation pressure according to Davis et al. (1991)
    # https://www.nist.gov/system/files/documents/calibrations/metv29i1p67-2.pdf
    A = 1.2378847*10**(-5)
    B = -1.9121316*10**(-2)
    C = 33.93711047
    D = -6.3431645*10**3
    p_sv = np.exp(A*T**2 + B*T + C + D/T) # psv is valiated with the paper
    x_w = h*f*p_sv/p
    return x_w

def c0_cramer(h, celsius, p=101325, x_c=0.0004):
    """speed of sound in air with humidity according to Cramer (1993)

    default value for the mole fraction of CO2 is 0.0004 (same as used by the national physics laboratory (NPL) calculator)

    This code uses the original work of Giacomo et al. (1982) to calculate the mole fraction of water vapor in the air.
    A more recent work often  referenced to obtain the mole fraction of water vapor is the work of Davis et al. (1991).

    :param h: relative humidity in [0,1]
    :param celsius: temperature in celsius
    :param p: atmospheric pressure in Pa (default is the standard pressure 101325 Pa)
    :param x_c: carbon dioxide mole fraction
    :return: speed of sound in m/s

    """
    a0 = 331.5024
    a1 = 0.603055
    a2 = -0.000528
    a3 = 51.471935
    a4 = 0.1495874
    a5 = -0.000782
    a6 = -1.82*10**(-7)
    a7 = 3.73*10**(-8)
    a8 = -2.93*10**(-10)
    a9 = -85.20931
    a10 = -0.228525
    a11 = 5.91*10**(-5)
    a12 = -2.835149
    a13 = -2.15*10**(-13)
    a14 = 29.179762
    a15 = 0.000486
    T = celsius + 273.15
    x_w = mole_fraction_of_water_vapor(h, T, p)
    return a0 + a1*celsius + a2*celsius**2 + (a3 + a4*celsius + a5*celsius**2)*x_w + \
         (a6 + a7*celsius + a8*celsius**2)*p + (a9 + a10*celsius + a11*celsius**2)*x_c + \
            a12*x_w**2 + a13*p**2 + a14*x_c**2 + a15*x_w*p*x_c


if __name__ == "__main__":
    temperature = oht20_sensor.read_temperature()
    humidity = oht20_sensor.read_humidity()
    speed_of_sound = c0_cramer(humidity, temperature)

    # Ensure the 'messungen' directory exists
    if os.path.exists('messungen'):
        # Get the current date and time
        date_time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Create the filename
        filename = f"messungen/temperature-{date_time_str}.csv"

        # Save data to CSV file unrounded
        with open(filename, 'w') as f:
            f.write("Temperature,Humidity,Speed_of_Sound\n")
            f.write(f"{temperature},{humidity},{speed_of_sound}\n")
            
        print("saved temp.")