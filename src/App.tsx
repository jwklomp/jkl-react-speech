import React, {useEffect, useState} from 'react';
import logo from './logo.svg';
import './App.css';
import * as speech from "@tensorflow-models/speech-commands"
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import {SpeechCommandRecognizer, SpeechCommandRecognizerResult} from "@tensorflow-models/speech-commands"


const App = () => {
// 1. Create model and action states
    const [model, setModel] = useState<SpeechCommandRecognizer | null>(null)
    const [action, setAction] = useState<String | null>(null)
    const [labels, setLabels] = useState<Array<String> | null>(null)

// 2. Create Recognizer

    const loadModel = async () => {
        await tf.setBackend('webgl')
        const recognizer = await speech.create("BROWSER_FFT") // BROWSER_FFT uses the browser's native Fourier transform.
        await recognizer.ensureModelLoaded();
        console.log(recognizer.wordLabels())
        setModel(recognizer)
        setLabels(recognizer.wordLabels())
    }

    useEffect(() => {
        loadModel().then(r => console.log("model loaded", r))
    }, []);

//
    const argMax = (arr: Array<number>) =>
        arr.map((x: number, i: number) => [x, i]).reduce((r: Array<number>, a: Array<number>) => (a[0] > r[0] ? a : r))[1];


// 3. Listen for Actions
    const recognizeCommands = async () => {
        if (model != null) {
            console.log('Listening for commands')
            // @ts-ignore
            model.listen((result: SpeechCommandRecognizerResult) => {
                // console.log(labels[argMax(Object.values(result.scores))])
                console.log(result.spectrogram)
                if (labels) {
                    setAction(labels[argMax(Object.values(result.scores))])
                }
            }, {includeSpectrogram: true, probabilityThreshold: 0.8})
            setTimeout(() => model.stopListening(), 10e3)
        }
    }

    return (
        <div className="App">
            <header className="App-header">
                <img src={logo} className="App-logo" alt="logo"/>
                <p>
                    Edit <code>src/App.js</code> and save to reload.
                </p>
                <button onClick={recognizeCommands}>Command</button>
                {action ? <div>{action}</div> : <div>No Action Detected</div>}
            </header>
        </div>
    );
}

export default App;
