

states = {
    500094 : ["H No 37 74/11/1, Plot No E-62, Survey No 218/1/3, JJ Nagar Ramakrishnapuram, "+
              "Malkajgiri Municipality and Mandal Ranga Reddy , Neredmet ,Hyderabad, 500094"],
    500015 : ["Plot No 6 S, Nos 60 6162/1, New Vasavi Colony, Second Venture Kakaguda, Village"+
                                      " Secunderabad Cantonment Area, Vikrampuri"],
    500039 : ["FF 02 &amp; 03, First Floor, DSL Virtue Mall, Ramanthapur Main Road IDA Uppal"],
    500079 : ["Shop No 17/1/383/N/80/31, Block No 1, Brindavan Colony Nagar"],
    500074 : ["Plot No 124 In Survey No 64, Ward No 3, Block No 8, Door No 3-11-22, Sri Enclave"+
                  ", Suvidha Arcade, RTC Colony, LB Nagar, Bahadurguda"],
    500072 : ["D#12-6-2/216, P#1 SY#577, Ground Floor, Kukatpally","MIG 161,"+
                                " Ground Floor, Road No 1, KPHB Colony"],
    500084 : ["Door No. 4-1382, Raja Rajeswara Nagar, Kondapur Village, "+
                                  "Serilingampally Mandal, Ranga Reddy"],
    500054 : ["Shop No 6/7, House No 60/9/9, MPR Complex , IDPL Y Junction, Chintal"],
    500062 : ["Victory Plaza, Plot No 11 Extension, ECIL X Main Road, Khushaiguda"],
    500036 : ["Shop No 16/10/27/105/7/4/1, Tirumala Pride, Main Road ,Malakpet"],
    500003 : ["Shop No 1,7 &amp; 396, Sandhu Apartment, Jain Estate, Kalasiguda, "+
    "Secunderabad ,Kalasiguda","PB 1549, M G Road, Kalasiguda, Secunderabad,MG Road"],
    500044 : ["House No 1/9/670/69, Ground Floor, SRT Quarters,Vidya Nagar, Adikmet"],
    500072 : ["MIG 42, Ground Floor, KPHB Phase 1, Dharma Reddy Colony ,Kukatpally"],
    500084 : ["Shop No 1/112/80,Kondapur<br>Hyderabad, 500084"],
    500026 : ["Building No, 3 &amp; 6,101/C/1J, H/228, Chiranjeevi Apartment,"+
    " West Marredpally Road, Boosareddy Guda, Aswini Colony , West Marredpally"],
    500044 : ["H No 2/2/647/3/A/3, Ground Floor, Shivam Road, Amberpet, "+
                          "APHB Colony, New Nallakunta New Nallakunta"],
    501218 : ["23/65/1, Madhura Nagar, Shamsahabad, Rangareddy,Shamsahbad"],
    500081 : ["H No 1-90/20/G/1, Sai Siddartha Complex, Metro Pillar No 1760, "+
    "Main Road , Madhapur","Shop No 64, SN DK1A, Level 4, Food Court, Inorbit Mall"+
              " , APIIC Software Layout, Madhapur"],
    500007 : ["Shop No 1/4/53/1, Street No 8 ,Habsiguda"],
    500015 : ["Plot No 57, Ground Floor, Part A1, P &amp; T Colony ,Trimulgherry"],
    500029 : ["Shop No 3, 6/564/5 To 8, Circle, Himayat Nagar, AP State"+
                            " Housing Board,Himayat Nagar"],
    500038 : ["262/3RT, Ground Floor, MN Reddy Classic, SR Nagar"],
    500060 : ["Shop No 13/4/30, Ground Floor,Dilsukhnagar, Vikas Nagar"],
    500070 : ["Ground Floor, Shop No 1, Old Court Building, Nagarjun Sagar"+
                                " Road, Omkar Nagar,Omkar Nagar"],
    500050 : ["2-88, National Highway 65, Gangaram, Chanda Nagar BHEL"],
    500049 : ["Shop No G2, Ground Floor, Kalyan Tulasi Ram Chambers, Madeenaguda",
                      "Shop No FF 101, First Floor, GSM Mall, Madinaguda"],
    500031 : ["H No 1-57/19, Sri Ram Nagar, Botanical Garden Road, Kondapur"],
    500029 : ["Flat No 3-6-365, Y V Rao Mansion;105 Himayatnagar Road, "+
                                            "Gagan Mahal, Himayatnagar"],
    500059 : ["Shop No 2, Ground Floor, 17-1-376/3, 34 Khaja Estate, Shantosh Nagar"],
    500034 : ["MCH No 6/3/348/B/201A, Ground Floor, BKE Owners Society, Road 1"],
    500019 : ["Ground Floor, GHMC 7-7/205, Nallagandle, Near Reliance Digital , Nallagandle"],
    500008 : ["House No 8-1-284/OU/20, Ground Floor, OU Colony, Shaikpet",
                  "13-6-798/1/21, New Friends Colony, Langar Houz"],
    500010 : ["171 12, NR Temple, Alwal village, GHMC, Alwal Circle, "+
                                "Malkajgiri Mandal, Ranga Reddy"],
    500018 : ["House No 3-6-23/4, Plot No 297, Survey Numbers 137&amp;139, "+
    "Bagh Ameeri village, Balanagar Madal, Kukatpally Cricle GHMC, Ranga Reddy"],
    500036 : ["16-11-741/C/13/5, Ground Floor, Vijetha Royal Empire, "+
                          "Indira Nagar, Dilsukh Nagar"],
    500062 : ["House No 3-30/1, Plot No1, Sri Nagar Colony, Kapra Village &amp;"+
                            " Mandal, Medchal District"],
    500048 : ["Ground Floor, Baba Towers, Tilak Road, King Koti"],
    500067 : ["Plot No 6, Ground Floor, Ganesh Housing Colony, "+
                            "Suchitra X Roads, Quthbullapur"],
    500085 : ["Dno-16-2-M/14/G1, Ground Floor, Sai Plaza, Charma Reddy Colony, "+
                    "Phase II, Hydernagar Village, kukatpally"],
    500089 : ["Plot # 68A, Alkapur Road # 7, Puppal Guda"],
    500033 : ["Municipal No 8/2/293/82/A471, Road No 36"],
    500076 : ["EMR Complex No 4-1-91/6, Bhavani Nagar, Nacharam Main Road, Nacharam",
    "House No 3-498/1/1/3, Ground Floor, Plot No 3, Situated Mallapur"+
        " village, Uppal Mandal, Kapra Circle"]
}


def closest_value(input_list, input_value):
  difference = lambda input_list : abs(input_list - input_value)
  res = min(input_list, key=difference)
  return res
 
def get_address(pincode):
    val=closest_value(sorted(states.keys()),pincode)
    return states[val]
