'''
###########################################################################################
使用说明
------------------------
此程序用于处理天格载荷（MCU方案）所产生的dat或txt格式数据文件。

在命令行进入此程序文件所在的路径，输入以下命令:
python GRID-04_QuickProcess.py "数据文件存放的路径"
（如果系统里同时安装有python2版本，需使用 python3 GRID-04_QuickProcess.py "数据文件存放的路径"）
则在数据文件存放的路径下会创建一个"quick_output"文件夹，数据处理的结果会保存在里面。

如果希望处理单个数据文件，则命令改为：python GRID-04_QuickProcess.py "包含路径的文件名"

在命令末尾增加"（空格）1"，如：python GRID-04_QuickProcess.py "数据文件存放的路径" 1
则会直接显示出绘制出的图片。每当显示出的图片被关闭后，程序才会继续绘制下一张图片。
###########################################################################################
'''
import re
import struct
#import crc16
from crcmod import mkCrcFun
from binascii import unhexlify
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import sys

def read_raw(filename):
    print('\nProcessing ' + os.path.split(filename)[1])
    if os.path.splitext(filename)[1] == '.dat':
        f = open(filename, 'rb')
        raw_data = f.read()
        f.close()
    else:
        f = open(filename, 'r')
        str_data = f.read()
        f.close()
        data_list = []
        strs = str_data.split()
        cnt = 0
        for st in strs:
            cnt += 1
            if cnt % 1000000 == 0:
                print('%d bytes converted...'%cnt)
            data_list.append(int(st, 16).to_bytes(1, byteorder='big'))
        raw_data = b''.join(data_list)
    sci, tel = extract_data(raw_data)
    return sci, tel

def find_pattern_pos(sdata, pattern):
    pos = []
    length = []
    for m in pattern.finditer(sdata):
        pos.append(m.start())
        length.append(m.end() - m.start())
    pos = np.array(pos)
    length = np.array(length)
    return pos, length

def check_crc16(sdata, scrc16):
    crc_in_data = struct.unpack('>H', scrc16[0:2])[0]
    calculated_crc = crc16xmodem(sdata[0:])
    if crc_in_data == calculated_crc:
        return True
    else:
        return False

def crc16xmodem(s):
    crc16 = mkCrcFun(0x11021, rev=False, initCrc=0x0000, xorOut=0x0000)
    return crc16(s)

def extract_packs(raw_data, pack_type):
    if pack_type == 'sci':
        pattern = re.compile(b'\xAA\xBB\xCC.{504}\xDD\xEE\xFF',re.S)  # re.S: '.' matches all, the length of pattern is 510 bytes
        pack_len = 512
    else: # tel
        pattern = re.compile(b'\x12\x34\x56.{490}\x78\x9A\xBC',re.S)  # re.S: '.' matches all, the length of pattern is 496 bytes
        pack_len = 498
    pack_pos, _ = find_pattern_pos(raw_data, pattern)
    
    total_pack_num = len(pack_pos)
    correct_pack_num = 0
    crc_error_pack_num = 0

    packages = []
    crc_error_data = []
    correct_data = []

    print('### %s packages ###'%(pack_type))
    print('  Number of packages:     {:>6d}'.format(total_pack_num))

    for i in range(total_pack_num):
        packages.append(raw_data[pack_pos[i]:pack_pos[i] + pack_len])
    
    for package in packages:
        if check_crc16(package[0 : pack_len - 2],package[pack_len - 2 : pack_len]):
            correct_pack_num += 1
            correct_data.append(package)
        else:
            crc_error_pack_num += 1
            crc_error_data.append(package)
    
    print('  Correct packages:       {:>6d}'.format(correct_pack_num))
    print('  CRC error packages:     {:>6d}'.format(crc_error_pack_num))

    return  correct_data, crc_error_data

def extract_data(raw_data):
    sci_pack, sci_pack_err = extract_packs(raw_data, 'sci')
    tel_pack, tel_pack_err = extract_packs(raw_data, 'tel')

    total_sci_num = len(sci_pack)
    print('### SCI packages ###')
    print('  Number of SCI packages: {:>6d}'.format(total_sci_num))

    channel =     []
    usc =         []  # 8byte uS count, not UTC time
    adc_value =   []
    utc =         []  # only the first event in a package contains UTC time
    pps =         []  # only the first event in a package contains PPS count
    valid_count = []  # only once at the last of the package
    lost_count =  []  # only once at the last of the package
   
    for package in sci_pack:
        channel.    append(struct.unpack('>B', package[  3 :   4])[0])
        usc.        append(struct.unpack('>Q', package[  4 :  12])[0])
        adc_value.  append(struct.unpack('>H', package[ 12 :  14])[0])
        utc.        append(struct.unpack('>L', package[ 14 :  18])[0])
        pps.        append(struct.unpack('>Q', package[ 18 :  26])[0])            
        for j in range(0,43):
            channel.  append(struct.unpack('>B', package[11 * j + 26 : 11 * j + 27])[0])
            usc.      append(struct.unpack('>Q', package[11 * j + 27 : 11 * j + 35])[0])
            adc_value.append(struct.unpack('>H', package[11 * j + 35 : 11 * j + 37])[0])
        valid_count.append(struct.unpack('>L', package[499 : 503])[0])
        lost_count. append(struct.unpack('>L', package[503 : 507])[0])

    sci = {
            'utc':         np.array(utc,         dtype=np.uint32),
            'usc':         np.array(usc,         dtype=np.uint64),
            'pps':         np.array(pps,         dtype=np.uint32),
            'channel':     np.array(channel,     dtype=np.uint16),
            'adc_value':   np.array(adc_value,   dtype=np.uint16),
            'valid_count': np.array(valid_count, dtype=np.uint32),
            'lost_count':  np.array(lost_count,  dtype=np.uint32),
        }


    total_tel_num = len(tel_pack)
    print('### TEL packages ###')
    print('  Number of TEL packages:  {:>6d}'.format(total_tel_num))

    utc_tel =        []
    pps_tel =        []
    usc_tel =        []
    sipm_temp0 = []
    sipm_temp1 = []
    sipm_temp2 = []
    sipm_temp3 = []
    adc_temp0 =  []
    adc_temp1 =  []
    adc_temp2 =  []
    adc_temp3 =  []
    vmon0 =      []
    vmon1 =      []
    vmon2 =      []
    vmon3 =      []
    imon0 =      []
    imon1 =      []
    imon2 =      []
    imon3 =      []
    mcu_temp =   []
    pps_utc =    []
    usc_pps =    []  # 8byte uS count, not UTC time

    for package in tel_pack:
        for j in range(7):
            utc_tel.   append(struct.unpack('>L', package[70 * j +  3 : 70 * j +  7])[0])
            pps_tel.   append(struct.unpack('>Q', package[70 * j +  7 : 70 * j + 15])[0])
            usc_tel.   append(struct.unpack('>Q', package[70 * j + 15 : 70 * j + 23])[0])
            t = struct.unpack('>H', package[70 * j + 23 : 70 * j + 25])[0]
            sipm_temp0.append(t if t < (1<<11) else -((1 << 12) - t + 1))
            t = struct.unpack('>H', package[70 * j + 25 : 70 * j + 27])[0]
            sipm_temp1.append(t if t < (1<<11) else -((1 << 12) - t + 1))
            t = struct.unpack('>H', package[70 * j + 27 : 70 * j + 29])[0]
            sipm_temp2.append(t if t < (1<<11) else -((1 << 12) - t + 1))
            t = struct.unpack('>H', package[70 * j + 29 : 70 * j + 31])[0]
            sipm_temp3.append(t if t < (1<<11) else -((1 << 12) - t + 1))
            t = struct.unpack('>H', package[70 * j + 31 : 70 * j + 33])[0]
            adc_temp0. append(t if t < (1<<11) else -((1 << 12) - t + 1))
            t = struct.unpack('>H', package[70 * j + 33 : 70 * j + 35])[0]
            adc_temp1. append(t if t < (1<<11) else -((1 << 12) - t + 1))
            t = struct.unpack('>H', package[70 * j + 35 : 70 * j + 37])[0]
            adc_temp2. append(t if t < (1<<11) else -((1 << 12) - t + 1))
            t = struct.unpack('>H', package[70 * j + 37 : 70 * j + 39])[0]
            adc_temp3. append(t if t < (1<<11) else -((1 << 12) - t + 1))
            vmon0.     append(struct.unpack('>H', package[70 * j + 39 : 70 * j + 41])[0])
            vmon1.     append(struct.unpack('>H', package[70 * j + 41 : 70 * j + 43])[0])
            vmon2.     append(struct.unpack('>H', package[70 * j + 43 : 70 * j + 45])[0])
            vmon3.     append(struct.unpack('>H', package[70 * j + 45 : 70 * j + 47])[0])
            imon0.     append(struct.unpack('>H', package[70 * j + 47 : 70 * j + 49])[0])
            imon1.     append(struct.unpack('>H', package[70 * j + 49 : 70 * j + 51])[0])
            imon2.     append(struct.unpack('>H', package[70 * j + 51 : 70 * j + 53])[0])
            imon3.     append(struct.unpack('>H', package[70 * j + 53 : 70 * j + 55])[0])
            mcu_temp.  append(struct.unpack('>H', package[70 * j + 55 : 70 * j + 57])[0])
            pps_utc.   append(struct.unpack('>Q', package[70 * j + 57 : 70 * j + 65])[0])
            usc_pps.   append(struct.unpack('>Q', package[70 * j + 65 : 70 * j + 73])[0])
    tel = {}
    tel['utc'] = np.array(utc_tel, dtype=np.uint32)
    tel['pps'] = np.array(pps_tel, dtype=np.uint16)
    tel['usc'] = np.array(usc_tel, dtype=np.uint64)
    tel['sipm_temp0'] = np.array(sipm_temp0, dtype=np.int16)
    tel['sipm_temp1'] = np.array(sipm_temp1, dtype=np.int16)
    tel['sipm_temp2'] = np.array(sipm_temp2, dtype=np.int16)
    tel['sipm_temp3'] = np.array(sipm_temp3, dtype=np.int16)
    tel['adc_temp0'] = np.array(adc_temp0, dtype=np.int16)
    tel['adc_temp1'] = np.array(adc_temp1, dtype=np.int16)
    tel['adc_temp2'] = np.array(adc_temp2, dtype=np.int16)
    tel['adc_temp3'] = np.array(adc_temp3, dtype=np.int16)
    tel['vmon0'] = np.array(vmon0, dtype=np.uint16)
    tel['vmon1'] = np.array(vmon1, dtype=np.uint16)
    tel['vmon2'] = np.array(vmon2, dtype=np.uint16)
    tel['vmon3'] = np.array(vmon3, dtype=np.uint16)
    tel['imon0'] = np.array(imon0, dtype=np.uint16)
    tel['imon1'] = np.array(imon1, dtype=np.uint16)
    tel['imon2'] = np.array(imon2, dtype=np.uint16)
    tel['imon3'] = np.array(imon3, dtype=np.uint16)
    tel['mcu_temp'] = np.array(mcu_temp, dtype=np.uint16)
    tel['pps_utc'] = np.array(pps_utc, dtype=np.uint16)
    tel['usc_pps'] = np.array(usc_pps, dtype=np.uint64)

    tel = cal_tel_value(tel)
    return sci, tel

def cal_tel_value(tel):
    num = len(tel['utc'])
    sipm_temp_C0 = np.zeros(num, dtype=np.float64)
    sipm_temp_C1 = np.zeros(num, dtype=np.float64)
    sipm_temp_C2 = np.zeros(num, dtype=np.float64)
    sipm_temp_C3 = np.zeros(num, dtype=np.float64)
    adc_temp_C0 = np.zeros(num, dtype=np.float64)
    adc_temp_C1 = np.zeros(num, dtype=np.float64)
    adc_temp_C2 = np.zeros(num, dtype=np.float64)
    adc_temp_C3 = np.zeros(num, dtype=np.float64)
    vmon_V0 = np.zeros(num, dtype=np.float64)
    vmon_V1 = np.zeros(num, dtype=np.float64)
    vmon_V2 = np.zeros(num, dtype=np.float64)
    vmon_V3 = np.zeros(num, dtype=np.float64)
    imon_uA0 = np.zeros(num, dtype=np.float64)
    imon_uA1 = np.zeros(num, dtype=np.float64)
    imon_uA2 = np.zeros(num, dtype=np.float64)
    imon_uA3 = np.zeros(num, dtype=np.float64)
    bias0 = np.zeros(num, dtype=np.float64)
    bias1 = np.zeros(num, dtype=np.float64)
    bias2 = np.zeros(num, dtype=np.float64)
    bias3 = np.zeros(num, dtype=np.float64)
    for i in range(num):
        sipm_temp_C0[i] = (tel['sipm_temp0'][i] - 4096) / 16.0 if tel['sipm_temp0'][i] > 2048 else tel['sipm_temp0'][i] / 16.0
        sipm_temp_C1[i] = (tel['sipm_temp1'][i] - 4096) / 16.0 if tel['sipm_temp1'][i] > 2048 else tel['sipm_temp1'][i] / 16.0
        sipm_temp_C2[i] = (tel['sipm_temp2'][i] - 4096) / 16.0 if tel['sipm_temp2'][i] > 2048 else tel['sipm_temp2'][i] / 16.0
        sipm_temp_C3[i] = (tel['sipm_temp3'][i] - 4096) / 16.0 if tel['sipm_temp3'][i] > 2048 else tel['sipm_temp3'][i] / 16.0
        adc_temp_C0[i] = (tel['adc_temp0'][i] - 4096) / 16.0 if tel['adc_temp0'][i] > 2048 else tel['adc_temp0'][i] / 16.0
        adc_temp_C1[i] = (tel['adc_temp1'][i] - 4096) / 16.0 if tel['adc_temp1'][i] > 2048 else tel['adc_temp1'][i] / 16.0
        adc_temp_C2[i] = (tel['adc_temp2'][i] - 4096) / 16.0 if tel['adc_temp2'][i] > 2048 else tel['adc_temp2'][i] / 16.0
        adc_temp_C3[i] = (tel['adc_temp3'][i] - 4096) / 16.0 if tel['adc_temp3'][i] > 2048 else tel['adc_temp3'][i] / 16.0
        vmon_V0[i] = tel['vmon0'][i] / 4096.0 * 3.3 * 11.0
        vmon_V1[i] = tel['vmon1'][i] / 4096.0 * 3.3 * 11.0
        vmon_V2[i] = tel['vmon2'][i] / 4096.0 * 3.3 * 11.0
        vmon_V3[i] = tel['vmon3'][i] / 4096.0 * 3.3 * 11.0
        imon_uA0[i] = tel['imon0'][i] / 4096.0 * 3.3 / 1.0
        imon_uA1[i] = tel['imon1'][i] / 4096.0 * 3.3 / 1.0
        imon_uA2[i] = tel['imon2'][i] / 4096.0 * 3.3 / 1.0
        imon_uA3[i] = tel['imon3'][i] / 4096.0 * 3.3 / 1.0
        bias0[i] = vmon_V0[i] - imon_uA0[i] * 1.1
        bias1[i] = vmon_V1[i] - imon_uA1[i] * 1.1
        bias2[i] = vmon_V2[i] - imon_uA2[i] * 1.1
        bias3[i] = vmon_V3[i] - imon_uA3[i] * 1.1
    tel['sipm_temp_C0'] = sipm_temp_C0
    tel['sipm_temp_C1'] = sipm_temp_C1
    tel['sipm_temp_C2'] = sipm_temp_C2
    tel['sipm_temp_C3'] = sipm_temp_C3
    tel['adc_temp_C0'] = adc_temp_C0
    tel['adc_temp_C1'] = adc_temp_C1
    tel['adc_temp_C2'] = adc_temp_C2
    tel['adc_temp_C3'] = adc_temp_C3
    tel['vmon_V0'] = vmon_V0
    tel['vmon_V1'] = vmon_V1
    tel['vmon_V2'] = vmon_V2
    tel['vmon_V3'] = vmon_V3
    tel['imon_uA0'] = imon_uA0
    tel['imon_uA1'] = imon_uA1
    tel['imon_uA2'] = imon_uA2
    tel['imon_uA3'] = imon_uA3
    tel['bias0'] = bias0
    tel['bias1'] = bias1
    tel['bias2'] = bias2
    tel['bias3'] = bias3
    return tel

def save_sci_csv(sci, pfilename):
    path = os.path.split(pfilename)[0]
    filename = os.path.split(pfilename)[1]
    output_path = os.path.join(path, 'quick_output')
    fname = os.path.splitext(filename)[0]
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(os.path.join(output_path, fname)):
        os.mkdir(os.path.join(output_path, fname))
    with open(os.path.join(output_path, fname, 'sci_' + fname + '.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pack_index', 'utc', 'usc', 'pps', 'channel',
                         'adc_value', 'valid_count', 'lost_count'])
        for i in range(len(sci['utc'])):
            for j in range(44):
                writer.writerow([i, sci['utc'][i], sci['usc'][44 * i + j], sci['pps'][i],
                                 sci['channel'][44 * i + j], sci['adc_value'][44 * i + j],
                                 sci['valid_count'][i], sci['lost_count'][i]])

def save_tel_csv(tel, pfilename):
    path = os.path.split(pfilename)[0]
    filename = os.path.split(pfilename)[1]
    output_path = os.path.join(path, 'quick_output')
    fname = os.path.splitext(filename)[0]
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(os.path.join(output_path, fname)):
        os.mkdir(os.path.join(output_path, fname))
    key = tel.keys()
    title = ['pack_index']
    for k in key:
        title.append(k)
    with open(os.path.join(output_path, fname, 'tel_' + fname + '.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(title)
        for i in range(len(tel['utc'])):
            new_row = [i]
            for k in key:
                new_row.append(tel[k][i])
            writer.writerow(new_row)

def plot_spec(sci, pfilename, show_flag):
    size = 8
    xmin = 0
    xmax = 10000
    ROIleft = 0
    ROIright = 65535
    path = os.path.split(pfilename)[0]
    filename = os.path.split(pfilename)[1]
    output_path = os.path.join(path, 'quick_output')
    fname = os.path.splitext(filename)[0]
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(os.path.join(output_path, fname)):
        os.mkdir(os.path.join(output_path, fname))
    adc0 = sci['adc_value'][sci['channel'] == 0]
    adc1 = sci['adc_value'][sci['channel'] == 1]
    adc2 = sci['adc_value'][sci['channel'] == 2]
    adc3 = sci['adc_value'][sci['channel'] == 3]
    adc_4ch = [adc0, adc1, adc2, adc3]
    fig = plt.figure(figsize=(10, 8))
    axs = fig.subplots(4,1)
    for i, (ax, adc) in enumerate(zip(axs, adc_4ch)):
        hist, bin_edge = np.histogram(adc, bins = 200, range=(xmin, xmax))
        ax.stairs(hist, bin_edge, label=f'ch{i}')
        # ax.hist(adc, bins=np.append(np.array(range(4096))*16, 65536), label=f'ch{i}', histtype='step')
        ax.axvline(ROIleft,c='r',label='left='+str(ROIleft))
        ax.axvline(ROIright,c='r',label='right='+str(ROIright))
        ax.text(0, 0, 'ROIcount='+str(len(adc[(adc>ROIleft)&(adc<ROIright)])))

        ax.set_ylabel('count', fontsize=size)
        ax.legend(fontsize=size)
        ax.grid()
        ax.set_xlim(xmin, xmax)
        ax.tick_params(axis='both', which='major', labelsize=size)
        yticks1 = ax.yaxis.get_major_ticks()
        yticks1[-1].label1.set_visible(False)
        print(len(adc[(adc>ROIleft)&(adc<ROIright)]), end=' ')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    fig.align_ylabels()
    plt.savefig(os.path.join(output_path, fname, '1_spectrum.png'))
    if show_flag:
        plt.show()
    plt.close()
    size = 16

def plot_light(sci, pfilename, show_flag):
    size = 20
    path = os.path.split(pfilename)[0]
    filename = os.path.split(pfilename)[1]
    output_path = os.path.join(path, 'quick_output')
    fname = os.path.splitext(filename)[0]
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(os.path.join(output_path, fname)):
        os.mkdir(os.path.join(output_path, fname))
    t0 = sci['usc'][sci['channel'] == 0] / 2.4e7
    t1 = sci['usc'][sci['channel'] == 1] / 2.4e7
    t2 = sci['usc'][sci['channel'] == 2] / 2.4e7
    t3 = sci['usc'][sci['channel'] == 3] / 2.4e7
    t_all = sci['usc'] / 2.4e7
    if len(t_all) > 0:
        t_bins = np.array(range(0, 10 * int(np.ceil(np.max(t_all))+10))) * 0.1
    else:
        t_bins = [-1, 1]
    fig = plt.figure(figsize=(10, 8))
    plt.subplot(4, 1, 1)
    plt.hist(t0, bins=t_bins, label='ch0', histtype='step')
    # plt.title('Light Curve of ' + filename)
    plt.ylabel('cp 0.1 s', fontsize=size)
    plt.legend(fontsize=size)
    plt.grid()
    # plt.ylim(0,75)
    plt.xticks(visible=False)
    ax1 = plt.gca()
    ax1.tick_params(axis='both', which='major', labelsize=size)
    plt.subplot(4, 1, 2)
    plt.hist(t1, bins=t_bins, label='ch1', histtype='step')
    plt.ylabel('cp 0.1 s', fontsize=size)
    plt.legend(fontsize=size)
    plt.grid()
    # plt.ylim(0,75)
    plt.xticks(visible=False)
    ax2 = plt.gca()
    ax2.tick_params(axis='both', which='major', labelsize=size)
    plt.subplot(4, 1, 3)
    plt.hist(t2, bins=t_bins, label='ch2', histtype='step')
    plt.ylabel('cp 0.1 s', fontsize=size)
    plt.legend(fontsize=size)
    plt.grid()
    # plt.ylim(0,75)
    plt.xticks(visible=False)
    ax3 = plt.gca()
    ax3.tick_params(axis='both', which='major', labelsize=size)
    plt.subplot(4, 1, 4)
    plt.hist(t3, bins=t_bins, label='ch3', histtype='step')
    plt.xlabel('time since startup (s)', fontsize=size)
    plt.ylabel('cp 0.1 s', fontsize=size)
    plt.legend(fontsize=size)
    plt.grid()
    # plt.ylim(0,75)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    ax4 = plt.gca()
    ax4.tick_params(axis='both', which='major', labelsize=size)
    # yticks1 = ax1.yaxis.get_major_ticks()
    # yticks1[0].label1.set_visible(False)
    # yticks1[1].label1.set_visible(False)
    # yticks1[3].label1.set_visible(False)
    # yticks1[4].label1.set_visible(False)
    # yticks2 = ax2.yaxis.get_major_ticks()
    # yticks2[0].label1.set_visible(False)
    # yticks2[1].label1.set_visible(False)
    # yticks2[3].label1.set_visible(False)
    # yticks2[4].label1.set_visible(False)
    # yticks3 = ax3.yaxis.get_major_ticks()
    # yticks3[-1].label1.set_visible(False)
    # yticks4 = ax4.yaxis.get_major_ticks()
    # yticks4[0].label1.set_visible(False)
    # yticks4[1].label1.set_visible(False)
    # yticks4[3].label1.set_visible(False)
    # yticks4[4].label1.set_visible(False)
    yticks1 = ax1.yaxis.get_major_ticks()
    yticks1[-1].label1.set_visible(False)
    yticks2 = ax2.yaxis.get_major_ticks()
    yticks2[-1].label1.set_visible(False)
    yticks3 = ax3.yaxis.get_major_ticks()
    yticks3[-1].label1.set_visible(False)
    yticks4 = ax4.yaxis.get_major_ticks()
    yticks4[-1].label1.set_visible(False)
    fig.align_ylabels()
    plt.savefig(os.path.join(output_path, fname, '2_light_curve.png'))
    if show_flag:
        plt.show()
    plt.close()
    size = 16

def plot_temp(tel, pfilename, show_flag):
    path = os.path.split(pfilename)[0]
    filename = os.path.split(pfilename)[1]
    output_path = os.path.join(path, 'quick_output')
    fname = os.path.splitext(filename)[0]
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(os.path.join(output_path, fname)):
        os.mkdir(os.path.join(output_path, fname))
    ti = tel['usc'] / 2.4e7
    plt.figure(figsize=(9, 6))
    plt.scatter(ti, tel['sipm_temp_C0'], label='ch0', s=12, marker='.', alpha=0.8)
    plt.scatter(ti, tel['sipm_temp_C1'], label='ch1', s=12, marker='.', alpha=0.8)
    plt.scatter(ti, tel['sipm_temp_C2'], label='ch2', s=12, marker='.', alpha=0.8)
    plt.scatter(ti, tel['sipm_temp_C3'], label='ch3', s=12, marker='.', alpha=0.8)
    plt.title('SiPM Temperature of '+ filename)
    plt.xlabel('time since startup (s)')
    plt.ylabel('temperature (°C)')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_path, fname, '3_SiPM_temperature.png'))
    if show_flag:
        plt.show()
    plt.close()

def plot_bias(tel, pfilename, show_flag):
    size = 20
    path = os.path.split(pfilename)[0]
    filename = os.path.split(pfilename)[1]
    output_path = os.path.join(path, 'quick_output')
    fname = os.path.splitext(filename)[0]
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(os.path.join(output_path, fname)):
        os.mkdir(os.path.join(output_path, fname))
    ti = tel['usc'] / 2.4e7
    plt.figure(figsize=(10, 7))
    plt.scatter(ti, tel['bias0'], label='ch0', s=12, marker='.', alpha=0.8)
    plt.scatter(ti, tel['bias1'], label='ch1', s=12, marker='.', alpha=0.8)
    plt.scatter(ti, tel['bias2'], label='ch2', s=12, marker='.', alpha=0.8)
    plt.scatter(ti, tel['bias3'], label='ch3', s=12, marker='.', alpha=0.8)
    # plt.title('SiPM Bias Voltage of '+ filename)
    plt.xlabel('time since startup (s)',fontsize=size)
    plt.ylabel('bias voltage (V)',fontsize=size)
    plt.legend(fontsize=size)
    plt.grid()
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=size)
    # plt.xlim(0,25)
    # plt.ylim(28.2,30)
    plt.savefig(os.path.join(output_path, fname, '4_SiPM_voltage.png'))
    if show_flag:
        plt.show()
    plt.close()
    size = 16

def plot_imon(tel, pfilename, show_flag):
    path = os.path.split(pfilename)[0]
    filename = os.path.split(pfilename)[1]
    output_path = os.path.join(path, 'quick_output')
    fname = os.path.splitext(filename)[0]
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(os.path.join(output_path, fname)):
        os.mkdir(os.path.join(output_path, fname))
    ti = tel['usc'] / 2.4e7
    plt.figure(figsize=(10, 7))
    plt.scatter(ti, tel['imon_uA0'], label='ch0', s=12, marker='.', alpha=0.8)
    plt.scatter(ti, tel['imon_uA1'], label='ch1', s=12, marker='.', alpha=0.8)
    plt.scatter(ti, tel['imon_uA2'], label='ch2', s=12, marker='.', alpha=0.8)
    plt.scatter(ti, tel['imon_uA3'], label='ch3', s=12, marker='.', alpha=0.8)
    # plt.title('SiPM Leakage Current of '+ filename)
    plt.xlabel('time since startup (s)',fontsize=size)
    plt.ylabel('leakage current (mA)',fontsize=size)
    plt.legend(fontsize=size)
    plt.grid()
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=size)
    # plt.ylim(0.03,0.06)
    plt.savefig(os.path.join(output_path, fname, '5_SiPM_current.png'))
    if show_flag:
        plt.show()
    plt.close()

def plot_IV(tel, pfilename, show_flag):
    path = os.path.split(pfilename)[0]
    filename = os.path.split(pfilename)[1]
    output_path = os.path.join(path, 'quick_output')
    fname = os.path.splitext(filename)[0]
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(os.path.join(output_path, fname)):
        os.mkdir(os.path.join(output_path, fname))
    plt.figure(figsize=(9, 6))

    ti = tel['usc'] / 2.4e7
    bias0 = tel['bias0'][ti < 50]
    imon_uA0 = tel['imon_uA0'][ti < 50]
    bias1 = tel['bias1'][ti < 50]
    imon_uA1 = tel['imon_uA1'][ti < 50]
    bias2 = tel['bias2'][ti < 50]
    imon_uA2 = tel['imon_uA2'][ti < 50]
    bias3 = tel['bias3'][ti < 50]
    imon_uA3 = tel['imon_uA3'][ti < 50]
    plt.scatter(bias0, imon_uA0, label='ch0', s=12, marker='.', alpha=0.8)
    plt.scatter(bias1, imon_uA1, label='ch1', s=12, marker='.', alpha=0.8)
    plt.scatter(bias2, imon_uA2, label='ch2', s=12, marker='.', alpha=0.8)
    plt.scatter(bias3, imon_uA3, label='ch3', s=12, marker='.', alpha=0.8)
    # plt.scatter(tel['bias0'], tel['imon_uA0'], label='ch0', s=12, marker='.', alpha=0.8)
    # plt.scatter(tel['bias1'], tel['imon_uA1'], label='ch1', s=12, marker='.', alpha=0.8)
    # plt.scatter(tel['bias2'], tel['imon_uA2'], label='ch2', s=12, marker='.', alpha=0.8)
    # plt.scatter(tel['bias3'], tel['imon_uA3'], label='ch3', s=12, marker='.', alpha=0.8)

    # plt.title('SiPM I-V Relationship of '+ filename)
    plt.xlabel('bias voltage (V)',fontsize=size)
    plt.ylabel('leakage current (mA)',fontsize=size)
    plt.legend(fontsize=size)
    plt.grid()
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=size)
    plt.savefig(os.path.join(output_path, fname, '6_SiPM_IV.png'))
    if show_flag:
        plt.show()
    plt.close()

def plot_utc(tel, pfilename, show_flag):
    path = os.path.split(pfilename)[0]
    filename = os.path.split(pfilename)[1]
    output_path = os.path.join(path, 'quick_output')
    fname = os.path.splitext(filename)[0]
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(os.path.join(output_path, fname)):
        os.mkdir(os.path.join(output_path, fname))
    ti = tel['usc'] / 2.4e7
    plt.figure(figsize=(9, 6))
    plt.scatter(ti, tel['utc'], s=12, marker='.')
    plt.title('UTC Broadcast of '+ filename)
    plt.xlabel('time since startup (s)')
    plt.ylabel('UTC broadcast (s)')
    plt.grid()
    plt.savefig(os.path.join(output_path, fname, '7_utc.png'))
    if show_flag:
        plt.show()
    plt.close()

def plot_utc_nz(tel, pfilename, show_flag):
    path = os.path.split(pfilename)[0]
    filename = os.path.split(pfilename)[1]
    output_path = os.path.join(path, 'quick_output')
    fname = os.path.splitext(filename)[0]
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(os.path.join(output_path, fname)):
        os.mkdir(os.path.join(output_path, fname))
    ti = tel['usc'][tel['utc'] != 0] / 2.4e7
    utc = tel['utc'][tel['utc'] != 0]
    plt.figure(figsize=(9, 6))
    plt.scatter(ti, utc, s=12, marker='.')
    plt.title('Non-zero UTC Broadcast of '+ filename)
    plt.xlabel('time since startup (s)')
    plt.ylabel('UTC broadcast (s)')
    plt.grid()
    plt.savefig(os.path.join(output_path, fname, '8_utc_non-zero.png'))
    if show_flag:
        plt.show()
    plt.close()

def plot_pps(tel, pfilename, show_flag):
    path = os.path.split(pfilename)[0]
    filename = os.path.split(pfilename)[1]
    output_path = os.path.join(path, 'quick_output')
    fname = os.path.splitext(filename)[0]
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(os.path.join(output_path, fname)):
        os.mkdir(os.path.join(output_path, fname))
    ti = tel['usc'] / 2.4e7
    plt.figure(figsize=(9, 6))
    plt.scatter(ti, tel['pps'], s=12, marker='.')
    plt.title('PPS Count of '+ filename)
    plt.xlabel('time since startup (s)')
    plt.ylabel('PPS count')
    plt.grid()
    plt.savefig(os.path.join(output_path, fname, '9_pps.png'))
    if show_flag:
        plt.show()
    plt.close()

def plot_pps_hist(tel, pfilename, show_flag):
    path = os.path.split(pfilename)[0]
    filename = os.path.split(pfilename)[1]
    output_path = os.path.join(path, 'quick_output')
    fname = os.path.splitext(filename)[0]
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(os.path.join(output_path, fname)):
        os.mkdir(os.path.join(output_path, fname))
    dusc = tel['usc_pps'][1:] - tel['usc_pps'][:-1]
    uscbins = range(int(2e7-500), int(2e7+500), 20)
    plt.figure(figsize=(9, 6))
    plt.hist(dusc, bins=uscbins)
    plt.title('usCounts between Two PPS Pulses of '+ filename)
    plt.xlabel('usCounts between two PPS pulses')
    plt.ylabel('count')
    plt.grid()
    plt.savefig(os.path.join(output_path, fname, '10_usCount_between_pps.png'))
    if show_flag:
        plt.show()
    plt.close()

def plot_and_save(sci, tel, pfilename, show_flag):
    plot_spec(sci, pfilename, show_flag)
    plot_light(sci, pfilename, show_flag)
    plot_temp(tel, pfilename, show_flag)
    plot_bias(tel, pfilename, show_flag)
    plot_imon(tel, pfilename, show_flag)
    plot_IV(tel, pfilename, show_flag)
    plot_utc(tel, pfilename, show_flag)
    plot_utc_nz(tel, pfilename, show_flag)
    plot_pps(tel, pfilename, show_flag)
    plot_pps_hist(tel, pfilename, show_flag)
    save_sci_csv(sci, pfilename)
    save_tel_csv(tel, pfilename)


if __name__ == '__main__':
    size = 16
    show_flag = 0
    arg1 = sys.argv[1]
    if len(sys.argv) >= 3:
        show_flag = int(sys.argv[2])
    if os.path.isfile(arg1):
    #if True:
        pfilename = arg1
        #pfilename = './0824_tempcycle_high_test.txt'
        sci, tel = read_raw(pfilename)
        # sci['usc'] = sci['usc'] - 40*2.4e7
        # tel['usc'] = tel['usc'] - 40*2.4e7
        # print(sci['usc']/2.4e7)
        # print(pfilename)
        # print(len(sci['usc'])/(sci['usc'][-1]-sci['usc'][0])*2.4e7)
        # with open('.\\final_test\\four\\record.csv', 'a', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow([os.path.splitext(pfilename.split('\\')[-1])[0] , len(sci['usc'])/(sci['usc'][-1]-sci['usc'][0])*2.4e7])
        plot_and_save(sci, tel, pfilename, show_flag)
    elif os.path.isdir(arg1):
        files = os.listdir(arg1)
        for f in files:
            if os.path.splitext(f)[1] == '.dat' or os.path.splitext(f)[1] == '.txt':
                pfilename = os.path.join(arg1, f)
                sci, tel = read_raw(pfilename)
                plot_and_save(sci, tel, pfilename, show_flag)