# reComputer Industrial Series — Key Technical Highlights
Hihligted varient of Nvidia Jetson Platform which is used for the Deeplearning experiment is J4011. 
See some important features of it below:

## 🚀 Edge AI Performance
- Built around NVIDIA Jetson modules such as Xavier NX, Orin Nano, and Orin NX.  
- AI compute ranges roughly from **20 TOPS up to 100 TOPS**, enabling demanding edge‑AI workloads.  
- Ships with **JetPack 5.1.1**, providing CUDA, TensorRT, and the full NVIDIA software stack out of the box.  
- Suitable for applications like:
  - Video analytics  
  - Object detection and tracking  
  - Natural language processing  
  - Medical imaging  
  - Robotics and autonomous systems  
- Designed to support digital transformation across sectors including smart cities, industrial automation, security, and smart factories.

## 🌡️ Rugged Thermal Design
- Features a **passive heatsink** with a **fanless enclosure**.  
- Benefits of this design:
  - Reliable cooling without moving parts  
  - Reduced dust ingress and lower maintenance needs  
  - Silent operation  
  - Lower overall power consumption  
- Ideal for harsh or noise‑sensitive industrial environments.

## 🌐 Networking & Connectivity
- Equipped with **two RJ45 Gigabit Ethernet ports**:
  - One port supports **PoE PSE**, allowing it to power external devices such as IP cameras directly over Ethernet.  
  - The second port provides standard Gigabit networking for connection to switches, routers, or industrial networks.  
- PoE capability simplifies deployment in locations where separate power lines are difficult to install.

## 🧩 About it
The reComputer Industrial series combines rugged hardware, strong AI performance, and flexible connectivity, making it a 
reliable platform for deploying edge AI solutions in real‑world industrial and commercial environments. 
There are many other variants available, but under 1,000 euros, this one is truly excellent.

# Upgrade to JetPack 6.0 on reComputer J4011

This guide outlines the steps required to upgrade the reComputer J4011 to **JetPack 6.0** using the official BSP package.

---

## 1. Extract the BSP Package

Download the BSP package and extract it:

```bash
sudo tar -xvf mfi_recomputer-industrial-orin-nx-8g-j201-6.0-36.3.0-2025-09-23.tar.gz
```

## 2. Enter the Extracted BSP Directory
Navigate into the BSP folder:

```bash
cd mfi_recomputer-industrial-orin-j201/
```

## 3. Flash JetPack (L4T) to the reComputer J4011
Run the flashing script to install JetPack 6.0:

```bash
sudo ./tools/kernel_flash/l4t_initrd_flash.sh --flash-only --massflash 1 --network usb0 --showlogs
```




