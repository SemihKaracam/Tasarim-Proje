import React, { useEffect, useState } from 'react'
import "./pulse.css"
import io from "socket.io-client"
import { Link } from 'react-router-dom'
import axios from "axios"

// const socket = io.connect("http://localhost:4000")

const Pulse = () => {
    const [nabiz, setNabiz] = useState(0)
    const [predict, setPredict] = useState("")
    const [isShow,setIsShow] = useState(false)
    useEffect(() => {
        const getPulse = async () => {
          const res = await axios.get("http://192.168.43.23/get")
        //   console.log(res.data.pulse)
          setNabiz(res.data.pulse)
        }   
        setInterval(()=>{
            getPulse()
        },1000)
      }, [])
    // useEffect(() => {
    //     socket.on("pulse", (data) => {
    //         setNabiz(data)
    //         console.log(data)
    //     })
    //     socket.on("predict", (data) => {
    //         // setPredict(data)
    //         console.log(data)
    //     })
    // }, [socket])
    // useEffect(()=>{
    //     if(nabiz!=102){
    //         setIsShow(true)
    //     }else{
    //         setIsShow(false)
    //     }
    // },[nabiz])
    return (
        <div className='pulse-page'>
            <Link to="/" className="anasayfaLink">Ana sayfa</Link>
            <div class="heart-rate">
                <svg version="1.0" xmlns="http://www.w3.org/2000/svg" xmlnsXlink="http://www.w3.org/1999/xlink" x="0px" y="0px" width="150px" height="73px" viewBox="0 0 150 73" enableBackground="new 0 0 150 73" xmlSpace="preserve">
                    <polyline fill="none" stroke="#009B9E" strokeWidth="3" strokeMiterlimit="10" points="0,45.486 38.514,45.486 44.595,33.324 50.676,45.486 57.771,45.486 62.838,55.622 71.959,9 80.067,63.729 84.122,45.486 97.297,45.486 103.379,40.419 110.473,45.486 150,45.486"
                    />
                </svg>
                <div class="fade-in"></div>
                <div class="fade-out"></div>
            </div>
            <div>
                {
                    !isShow ? <h4>Parmağınızı sensöre yerleştiriniz</h4> :
                    <h3>{nabiz != 102 ? nabiz : 0}</h3>
                }
                {nabiz}
            </div>
        </div>
    )
}

export default Pulse