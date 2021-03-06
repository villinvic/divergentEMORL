GALE01_ini_template = """
[Gecko]

$Flash White on Successful L-Cancel [Dan Salvato]
C208D6A0 00000002
C00600E8 39E000D4
99E30564 00000000
C20C0148 0000000C
387F0488 89FE0564
2C0F00D4 41820008
48000048 39E00091
99FE0564 3DE0C200
91FE0518 91FE051C
91FE0520 91FE0524
3DE00000 91FE0528
91FE052C 91FE0530
3DE0C280 91FE0534
3DE0800C 61EF0150
7DE903A6 4E800420
60000000 00000000

$Speedhack no render [taukhan]
041A5020 4BFFFDC4 # disable rendering
04019860 4BFFFD9D # prevents polling alarm
C20198E8 00000003 # new polling hook
9421FFF8 3D808001
618C95FC 7D8803A6
4E800021 00000000
040198C4 38600001 # allows game to proceed without polling hook
041A4D64 38600001 # remove 1 frame lag

$Speedhack test [taukhan]
041A5020 4BFFFDC4 # disable rendering
04019860 4BFFFD9D # prevents polling alarm
C20198E8 00000003 # new polling hook
9421FFF8 3D808001
618C95FC 7D8803A6
4E800021 00000000
040198C4 38600002 # allows game to proceed without polling hook
041A4D64 38600001 # remove 1 frame lag

$Speedhack [taukhan]
04019860 4BFFFD9D # prevents polling alarm
C20198E8 00000003 # new polling hook
9421FFF8 3D808001
618C95FC 7D8803A6
4E800021 00000000
040198C4 38600002 # allows game to proceed without polling hook
041A4D64 38600001 # remove 1 frame lag

$DMA Read Before Poll [Fizzi, xpilot]
C20055F0 00000027 #Codes/EXITransferBuffer.asm
7C0802A6 90010004
9421FFB0 BE810008
7C7E1B78 7C9D2378
7CBF2B78 7FC3F378
7C9EEA14 2C1F0000
4182001C 7C0018AC
38630020 7C032000
4180FFF4 7C0004AC
4C00012C 38600001
38800000 3D808034
618C64C0 7D8903A6
4E800421 38600001
3D808034 618C6D80
7D8903A6 4E800421
38600001 38800000
38A00005 3D808034
618C6688 7D8903A6
4E800421 38600001
7FC4F378 7FA5EB78
7FE6FB78 38E00000
3D808034 618C5E60
7D8903A6 4E800421
38600001 3D808034
618C5F4C 7D8903A6
4E800421 38600001
3D808034 618C67B4
7D8903A6 4E800421
38600001 3D808034
618C6E74 7D8903A6
4E800421 38600001
3D808034 618C65CC
7D8903A6 4E800421
2C1F0000 4082001C
7C001BAC 38630020
7C032000 4180FFEC
7C0004AC 4C00012C
BA810008 80010054
38210050 7C0803A6
4E800020 00000000
C216D294 00000006 #Codes/IncrementFrameIndex.asm
987F0008 3C608048
80639D58 2C030000
40820010 3860FF85
906DB654 48000010
806DB654 38630001
906DB654 00000000
C2019608 00000005 #Codes/ReadAfterPoll.asm
9421FFF8 7C230B78
38800000 38A00000
3D808000 618C55F0
7D8903A6 4E800421
38600000 00000000

$Boot to match
041A45A0 3800000E

$Skip Memcard Prompt
041AF724 48000AF0

$Setup match
{match_code}

[Gecko_Enabled]
{enabled}                 
"""

all = [
    '$Setup match',
    '$Skip Memcard Prompt',
    '$Boot to match',
    '$DMA Read Before Poll',
    '$Speedhack',
    '$Speedhack no render',
    '$Flash White on Successful L-Cancel',
]

demo = [

    '$Setup match',
    '$Skip Memcard Prompt',
    '$Boot to match',
    '$Flash White on Successful L-Cancel',

]

training = [
    '$Skip Memcard Prompt',
    '$Speedhack no render',
    '$Boot to match',
    '$Setup match',
    '$DMA Read Before Poll'
]
