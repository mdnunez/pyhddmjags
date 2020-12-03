## JAGS and JAGS-WIENER installation steps on Ubuntu

#### 1. Install JAGS:

```bash
sudo apt-get update
sudo apt-get install jags
```

#### 2. Install precompiled JAGS-WIENER for debian. If this works go to step 5.

This step has been tested on Ubuntu 18.04 LTS, Ubuntu 20.04 LTS and Ubuntu 18.04 LTS for Windows Subsystem Linux (WSL). This step will likely work for other Ubuntu distributions and may work for some other debian distributions.

```bash
TEMP_DEB="$(mktemp)"
wget -O "$TEMP_DEB" "https://launchpad.net/~cidlab/+archive/ubuntu/jwm/+files/jags-wiener-module_1.1-5_amd64.deb"
sudo dpkg -i "$TEMP_DEB"
rm -f "$TEMP_DEB"
```

#### 3. If step 2 doesn't work, install dependencies for JAGS-WIENER

```bash
sudo apt-get install autoconf automake libtool g++
```

#### 4. If step 2 doesn't work and step 3 completed successfully, download and install JAGS-WIENER

```bash
cd ~
mkdir install
cd install
wget http://downloads.sourceforge.net/project/jags-wiener/JAGS-WIENER-MODULE-1.1.tar.gz
tar -zxvf JAGS-WIENER-MODULE-1.1.tar.gz
cd JAGS-WIENER-MODULE-1.1
./configure --prefix=/usr/lib/x86_64-linux-gnu
make
sudo make install
```

Note that the prefix in the 3rd to last line above may change based on the install location of JAGS. This is especially likely if this error occurs:

```
DWiener.h:5:10: fatal error: distribution/ScalarDist.h: No such file or directory
#include <distribution/ScalarDist.h>
```

Try this command to find the correct prefix in the terminal:

```bash
find /usr -type f -name dic.la
```

For instance, if the result is ”/usr/lib/local/JAGS/modules-4/dic.la” Then I would use the following prefix in the 3rd to last line:

```bash
./configure --prefix=/usr/lib/local
```

#### 5. Test installation (last three commands will be within JAGS terminal):

```bash
jags
```
```
load dic
load wiener
exit
```

A successful installation of JAGS should load the modules: basemod “ok”, bugs “ok”, and dic “ok”. A successful installation of JAGS-WIENER that is found by JAGS will load wiener “ok”.

#### 6. (Optional) Install pyjags and dependencies (after installing Anaconda Python):

```bash
sudo apt-get install pkg-config
pip install pyjags
```
