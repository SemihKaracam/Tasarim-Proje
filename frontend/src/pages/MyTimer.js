import { useEffect } from 'react';
import { useTimer } from 'react-timer-hook';

export default function MyTimer({ setIsModalOpen,isModalOpen,expiryTimestamp }) {
    const {
        totalSeconds,
        seconds,
        pause,
        resume,
        restart,
    } = useTimer({ expiryTimestamp, onExpire: () => console.warn('onExpire called') });

    useEffect(()=>{
        if(totalSeconds == 0){
            setIsModalOpen(true)
            const time = new Date();
            time.setSeconds(time.getSeconds() + 10);
            restart(time)
            pause()
        }
    },[totalSeconds])

    useEffect(()=>{
        if(!isModalOpen){
            resume()
        }
        console.log("degisiyor")
        console.log(totalSeconds)
    },[isModalOpen])

    return (
        <div style={{ textAlign: 'center' }}>
            {/* <h1>react-timer-hook </h1>
            <p>Timer Demo</p> */}
            <div style={{ fontSize: '32px' }}>
                {/* <span>{days}</span>:<span>{hours}</span>:<span>{minutes}</span>:<span>{seconds}</span> */}
                <span>{seconds}</span>
            </div>
            {/* <p>{isRunning ? 'Running' : 'Not running'}</p>
            <button onClick={start}>Start</button>
            <button onClick={pause}>Pause</button>
            <button onClick={resume}>Resume</button>
            <button onClick={() => {
                // Restarts to 5 minutes timer
                const time = new Date();
                time.setSeconds(time.getSeconds() + 10);
                restart(time)
            }}>Restart</button> */}
        </div>
    );
}

// export default function App() {
//     const time = new Date();
//     time.setSeconds(time.getSeconds() + 10); 
//     return (
//         <div>
//             <MyTimer expiryTimestamp={time} />
//         </div>
//     );
// }