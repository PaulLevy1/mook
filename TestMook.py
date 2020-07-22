import unittest
import mook

class MyTestCase(unittest.TestCase):
    def test_simulate(self):
       # recession = mook.HistoricalSimulation('SPY', '1/6/2007','11/6/2007', '11/18/2009', 1000,1)
        #r1 = recession.simulate()
        #roiBuySell = r1[0]
        #roiDuration1 = r1[1]
        test3 = mook.HistoricalSimulation('SPY', '7/13/2002', '1/1/2003', '5/8/2020', 1000,0)
        r2 = test3.simulate()
        roiBuy = r2[0]
        roiDuration2 = r2[1]

        self.assertEqual(roiDuration2 > 1,True)


if __name__ == '__main__':
    unittest.main()
