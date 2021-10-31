import React, {FormEvent, useState} from 'react';
import './App.css';
import * as speech from "@tensorflow-models/speech-commands"
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import {TransferSpeechCommandRecognizer} from "@tensorflow-models/speech-commands/dist/types";

const numberOfSamples = 8;

const enterWordsText = "Enter one or more actions (words), separated by comma and press start"
const noWordsFoundText = "No words were found, enter one or more actions (words), separated by comma and press start"
const wordsFoundText = "Input is valid. Press Make model to start generating the model. Words found: "

const App = () => {
// Create model and action states
    const [model, setModel] = useState<TransferSpeechCommandRecognizer | null>(null)
    const [commandsAsText, setCommandsAsText] = useState<string>("")
    const [commands, setCommands] = useState<Array<string>>([])

    const [actionText, setActionText] = useState<string>(enterWordsText)

    const handleChange = (e: FormEvent<HTMLInputElement>) => setCommandsAsText(e.currentTarget.value)

    const enterCommands = () => {
        const trimmed = commandsAsText.trim();
        if (trimmed.length > 0) {
            const _commands = trimmed.split(",")
            const trimmedArray = _commands.map(it => it.trim())
            setCommands(trimmedArray)
            setActionText(wordsFoundText + _commands)
        } else {
            setActionText(noWordsFoundText)
        }
    }

    const sleep = (delay: number) => new Promise((resolve) => setTimeout(resolve, delay))

    const getSound = async (recognizer: TransferSpeechCommandRecognizer, word: string, count: number) => {
        setActionText(`Proceed with entering the word: ${word}`);
        await sleep(5000)
        for (let i = 0; i < count; i += 1) {
            setActionText(`Say: ${word}`);
            await recognizer.collectExample(word);
        }
    }

    const makeModel = async () => {
        await tf.setBackend('webgl')
        const baseRecognizer = await speech.create("BROWSER_FFT") // BROWSER_FFT uses the browser's native Fourier transform.
        console.log('Model Loading...');
        await baseRecognizer.ensureModelLoaded();

        console.log('Building transfer recognizer...');
        const transferRecognizer = baseRecognizer.createTransfer("navigation");

        for (const command of commands) { // await does not work in forEach
            await getSound(transferRecognizer, command, numberOfSamples);
        }

        await getSound(transferRecognizer, '_background_noise_', numberOfSamples);
        console.log('Finished ', 'background: #ccff99; color: black');
        console.table(transferRecognizer.countExamples());
        console.log('Training... ', 'background: #ffcccc; color: black');
        setActionText(`Training...`);
        await transferRecognizer.train({
            epochs: 25,
            callback: {
                onEpochEnd: async (epoch, logs) => {
                    console.log(`Epoch ${epoch}: loss=${logs?.loss}, accuracy=${logs?.acc}`);
                },
            },
        });
        setModel(transferRecognizer)
        setActionText(`Training completed`);
    }

    const runModel = async () => {
        setActionText('Listening... ');
        if (model !== null) {
            await model.listen(
                // @ts-ignore
                result => {
                    const words = model.wordLabels();
                    // @ts-ignore
                    const scores = Array.from(result.scores).map((s, i) => ({
                        score: s,
                        word: words[i],
                    }));

                    scores.sort((s1, s2) => s2.score - s1.score);
                    setActionText(`Score for word '${scores[0].word}' = ${scores[0].score}`);
                },
                {probabilityThreshold: 0.75}
            );
            setTimeout(() => {
                model.stopListening();
                setActionText('Stopped listening');
            }, 20e3);
        }
    }

    const saveByteArray = (function () {
        let a = document.createElement("a");
        document.body.appendChild(a);
        return function (data: BlobPart[], name: string) {
            // @ts-ignore
            const blob = new Blob(data, {type: "octet/stream"}),
                url = window.URL.createObjectURL(blob);
            a.href = url;
            a.download = name;
            a.click();
            window.URL.revokeObjectURL(url);
        };
    }());


    const downloadModel = () => {
        if (model !== null) {
            const serialized = model.serializeExamples();
            console.log("serialized", serialized);
            saveByteArray([serialized], "trainedmodel.txt");
        }
    }

    return (
        <div className="App">
            <main>
                <div>
                    <h2>Transfer learning with Tensorflowjs</h2>
                    <p>{actionText}</p>
                </div>
                <div>
                    <input type="text" value={commandsAsText} onChange={handleChange}/>
                </div>
                <div>
                    <button onClick={enterCommands}>Start</button>
                    <button disabled={commands.length === 0} onClick={makeModel}>Make model</button>
                    <button disabled={model === null} onClick={runModel}>Run model</button>
                    <button disabled={model === null} onClick={downloadModel}>Download model</button>
                </div>
            </main>
        </div>
    );
}

export default App;
