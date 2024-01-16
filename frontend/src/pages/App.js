import '../style.css';
import { IoMdPulse } from "react-icons/io"
import { BsFillPersonVcardFill } from "react-icons/bs"
import { GiCigarette } from "react-icons/gi"
import { BiBody } from "react-icons/bi"
import ReactLoading from 'react-loading';
import { FaHeartbeat } from "react-icons/fa";
import { SlLogout } from "react-icons/sl";

import { useAuth } from "../contexts/AuthContext";

import io from "socket.io-client"
import React, { useEffect, useRef, useState } from 'react';
import axios from "axios"
import { useNavigate } from 'react-router-dom';
import MyTimer from './MyTimer';
import { useTime, useTimer } from 'react-timer-hook';
import { Modal } from 'bootstrap'


// const socket = io.connect("http://localhost:4000")


function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function App() {
  const intervalRef = useRef(null);
  const [nabiz, setNabiz] = useState(0)
  const [loading, setLoading] = useState(false)
  const { currentUser, setCurrentUser, isLoggedIn, setIsLoggedIn } = useAuth()
  const [risk, setRisk] = useState(0)
  const [isModalOpen, setIsModalOpen] = useState(false);

  const openModal = () => {
    setIsModalOpen(true);
  }
  const closeModal = () => {
    // clearInterval(intervalRef.current)
    setIsModalOpen(false)
  }

  // useEffect(() => {
  //   intervalRef.current = setInterval(() => {
  //     openModal();
  //   }, 1 * 10000);
  // }, [isModalOpen]);

  // useEffect(() => {
  //   var myModal = new bootstrap.Modal(document.getElementById('exampleModal'))
  //   if (isModalOpen) {
  //     myModal.show()
  //   } else if (!isModalOpen) {
  //     intervalRef.current = setInterval(() => {
  //       openModal()
  //     }, 1 * 10000)
  //     myModal.hide()
  //   }
  // }, [isModalOpen])



  // useEffect(() => {
  //   const getPulse = async () => {
  //     const res = await axios.get("http://192.168.43.23:80/get")
  //     console.log(res)
  //     setNabiz(res.data.pulse)
  //   }
  //   setInterval(() => {
  //     getPulse()
  //   }, 1000)
  // }, [])

  // const measureRisk = async () => {
  //   await axios.post("http://localhost:5000/measure", { id: currentUser.id, nabiz: 85 })
  // }
  const navigate = useNavigate()

  // useEffect(()=>{
  //   if(localStorage.getItem("user") == null){
  //     navigate("/login")
  //   }
  // },[])
  const measureRisk = async () => {
    setLoading(true)
    console.log('Hello');
    await sleep(500)
    const { id, mail, ad, soyad } = JSON.parse(localStorage.getItem("user"))
    // const res = await axios.post("http://localhost:5000/measure",{id:currentUser.id,mail:currentUser.mail,ad:currentUser.ad,soyad:currentUser.soyad})
    const res = await axios.post("http://localhost:5000/measure", { id, mail, ad, soyad })
    console.log(res)
    setLoading(false)
    setRisk(res.data)
    console.log("measure risk fonk")
  }

  const handleLogout = () => {
    setIsLoggedIn(false)
    setCurrentUser(null)
    localStorage.setItem("user", null)
    navigate("/login")
  }

  // const time = new Date();
  // time.setSeconds(time.getSeconds() + 10);


  // const { seconds } = useTime({ format: '12-hour' })


  //timer için gerekli yapı
  const expiryTimestamp = new Date();
  expiryTimestamp.setSeconds(expiryTimestamp.getSeconds() + 15);
  const {
    totalSeconds,
    seconds,
    pause,
    resume,
    restart,
  } = useTimer({ expiryTimestamp, onExpire: () => console.warn('onExpire called') });

  useEffect(() => {
    if (totalSeconds == 0) {
      setIsModalOpen(true)
      const time = new Date();
      time.setSeconds(time.getSeconds() + 15);
      restart(time)
      pause()
    }
  }, [totalSeconds])

  useEffect(() => {
    var myModal = new Modal(document.getElementById('exampleModal'))
    if (isModalOpen) {
      myModal.show()
      measureRisk()
      
    } else if (!isModalOpen) {
      resume()
      myModal.hide()
    }

  }, [isModalOpen])


  return (
    <div className='home-page'>
      <div className='navbar d-flex flex-row-reverse px-3'>
        <button onClick={handleLogout} className='btn btn-primary d-flex gap-2 cikis-btn px-3 py-2 align-items-center rounded'>
          <SlLogout size={24} />
          <span>
            Çıkış Yap
          </span>
        </button>
      </div>
      <div className="health-container">
        <button onClick={measureRisk} disabled type="button" class="btn btn-primary" data-bs-toggle="modal" data-bs-target="#exampleModal">
          {/* <MyTimer setIsModalOpen isModalOpen expiryTimestamp={time} /> */}
          <FaHeartbeat size={36} />
          <p className='fs-3'>
            Kalp Krizi Riskini Ölçme Geri Sayım
          </p>
          <span style={{ fontSize: '32px' }}>{seconds} saniye</span>
        </button>
        {/* Button trigger modal  */}
        {/* <button type="button" class="btn btn-primary" data-toggle="modal" data-target="#exampleModal">
        Launch demo modal
      </button>
    */}
        {/* Modal */}
        <div class="modal fade modal-lg" id="exampleModal" data-bs-backdrop="static" tabindex="-1" role="dialog" aria-labelledby="exampleModalLabel" aria-hidden="true">
          <div class="modal-dialog" role="document">
            <div class="modal-content">
              <div class="modal-header">
                <h4 class="modal-title" id="exampleModalLabel">Kalp krizi riski</h4>
                <button onClick={closeModal} type="button" class="close" data-bs-dismiss="modal" aria-label="Close">
                  <span aria-hidden="true">&times;</span>
                </button>
              </div>
              <div class="modal-body d-flex align-items-center justify-content-center">
                {
                  loading ?
                    (
                      <div className='d-flex flex-column align-items-center justify-content-center gap-4'>
                        <FaHeartbeat color='red' size={48} />
                        <p className='h3'>Kalp krizi riski ölçülüyor</p>
                        <ReactLoading type='spin' color='black' width={'32px'} height={'32px'} />
                      </div>
                    ) :
                  (
                    <div className='d-flex flex-column align-items-center justify-content-center gap-2'>
                      <FaHeartbeat color='red' size={48} />
                      <span class="result-text">Kalp krizi riskiniz</span>
                      {/* <h4>%{Math.floor(Math.random() * 101)}</h4> */}
                      {/* <h2>%{risk.toFixed(1)}</h2> */}
                      <h1 className='text-danger'>%{risk.toFixed(2)}</h1>
                    </div>
                  )
                }
              </div>
              <div class="modal-footer">
                <button onClick={closeModal} type="button" class="btn btn-secondary" data-bs-dismiss="modal">Kapat</button>
              </div>
            </div>
          </div>
        </div>
        <div className='ad-soyad shadow rounded grid-item p-3 position-relative'>
          <div className='mb-4 d-flex align-items-center justify-content-between'>
            <h4>Ad Soyad</h4>
            <BsFillPersonVcardFill size={24} />
          </div>
          {/* <p className='text-center grid-value'>{nabiz}</p> */}
          <p className='text-center grid-value'>{currentUser?.ad + ' ' + currentUser?.soyad}</p>
        </div>
        <div className='nabiz shadow rounded grid-item p-3 position-relative'>
          <div className='mb-4 d-flex align-items-center justify-content-between'>
            <h4>Nabız</h4>
            <IoMdPulse size={24} />
          </div>
          {/* <p className='text-center grid-value'>{nabiz}</p> */}
          <p className='text-center grid-value'>{nabiz}</p>
          {/* <p className='text-center grid-value'>{nabiz}</p> */}
          <div className='position-absolute bottom-2 end-2'>
            <ReactLoading type='spin' color='white' width={'32px'} height={'32px'} />
          </ div>
        </div>
        <div className='yas shadow rounded grid-item p-3'>
          <div className='mb-4 d-flex align-items-center justify-content-between'>
            <h4>Yaş</h4>
            <BsFillPersonVcardFill size={24} />
          </div>
          <p className='text-center grid-value'>{currentUser?.yas}</p>
        </div>
        <div className='sigara shadow rounded grid-item p-3'>
          <div className='mb-4 d-flex align-items-center justify-content-between'>
            <h4>Sigara</h4>
            <GiCigarette size={24} />
          </div>
          <p className='text-center grid-value'>Günde {currentUser?.sigaraGunde} adet</p>
        </div>
        <div className='bmi shadow rounded grid-item p-3'>
          <div className='mb-4 d-flex align-items-center justify-content-between'>
            <h4>Vücut kitle endeksi</h4>
            <BiBody size={24} />
          </div>
          <p className='text-center grid-value'>{currentUser?.vki}</p>
        </div>
        <div className='kanbasinci shadow rounded grid-item p-3'>
          <div className='mb-4 d-flex align-items-center justify-content-between'>
            <h4>Kan basıncı ilacı</h4>
            <BiBody size={24} />
          </div>
          <p className='text-center grid-value'>{currentUser?.kanbasinci == 0 ? 'Hayır' : 'Evet'}</p>
        </div>
        <div className='felc shadow rounded grid-item p-3'>
          <div className='mb-4 d-flex align-items-center justify-content-between'>
            <h4>Geçmişte felç geçirdi mi</h4>
            <BiBody size={24} />
          </div>
          <p className='text-center grid-value'>{currentUser?.felc == 0 ? 'Hayır' : 'Evet'}</p>
        </div>
        <div className='hipertansiyon shadow rounded grid-item p-3'>
          <div className='mb-4 d-flex align-items-center justify-content-between'>
            <h4>Hipertansiyon</h4>
            <BiBody size={24} />
          </div>
          <p className='text-center grid-value'>{currentUser?.hipertansiyon == 0 ? 'Hayır' : 'Evet'}</p>
        </div>
        <div className='diyabet shadow rounded grid-item p-3'>
          <div className='mb-4 d-flex align-items-center justify-content-between'>
            <h4>Diyabet</h4>
            <BiBody size={24} />
          </div>
          <p className='text-center grid-value'>{currentUser?.diyabet == 0 ? 'Hayır' : 'Evet'}</p>
        </div>
        <div className='cinsiyet shadow rounded grid-item p-3'>
          <div className='mb-4 d-flex align-items-center justify-content-between'>
            <h4>Cinsiyet</h4>
            <BiBody size={24} />
          </div>
          <p className='text-center grid-value'>{currentUser?.cinsiyet == 0 ? 'Kadın' : 'Erkek'}</p>
        </div>
      </div>
    </div>
  );
}

export default App;

