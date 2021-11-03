import React, {FormEvent, useState} from 'react';
import './App.css';
import * as speech from "@tensorflow-models/speech-commands"
import {SpeechCommandRecognizer} from "@tensorflow-models/speech-commands"
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import {TransferSpeechCommandRecognizer} from "@tensorflow-models/speech-commands/dist/types";

const numberOfSamples = 8;

const enterWordsText = "Enter one or more actions (words), separated by comma and press start"
const noWordsFoundText = "No words were found, enter one or more actions (words), separated by comma and press start"
const wordsFoundText = "Input is valid. Press Make model to start generating the model. Words found: "

const BACKGROUND_NOISE_TAG = speech.BACKGROUND_NOISE_TAG;

const App = () => {
// Create model and action states
    const [transferRecognizer, setTransferRecognizer] = useState<TransferSpeechCommandRecognizer | null>(null)
    const [commandsAsText, setCommandsAsText] = useState<string>("")
    const [commands, setCommands] = useState<Array<string>>([])

    const [actionText, setActionText] = useState<string>(enterWordsText)

    const fileInput = React.createRef<HTMLInputElement>();

    const handleChange = (e: FormEvent<HTMLInputElement>) => setCommandsAsText(e.currentTarget.value)

    const enterCommands = () => {
        const trimmed = commandsAsText.trim();
        if (trimmed.length > 0) {
            const _commands = trimmed.split(",")
            const trimmedArray = _commands.map(it => it.trim()).sort()
            setCommands(trimmedArray)
            setActionText(wordsFoundText + _commands)
        } else {
            setActionText(noWordsFoundText)
        }
    }

    const sleep = (delay: number) => new Promise((resolve) => setTimeout(resolve, delay))

    const getSound = async (recognizer: TransferSpeechCommandRecognizer, word: string, count: number) => {
        setActionText(`Proceed with entering the word: ${word}`);
        await sleep(3000)
        for (let i = 0; i < count; i += 1) {
            setActionText(`Say: ${word}`);
            await recognizer.collectExample(word);
        }
    }

    async function makeBaseRecognizer() {
        await tf.setBackend('webgl')
        // speech commands vocabulary feature, not useful for your models
        const baseRecognizer = await speech.create(
            "BROWSER_FFT" // BROWSER_FFT uses the browser's native Fourier transform.
            , undefined) // speech commands vocabulary feature, not useful for your models
        console.log('Model Loading...');
        await baseRecognizer.ensureModelLoaded();
        return baseRecognizer;
    }

    const makeModel = async () => {
        const baseRecognizer = await makeBaseRecognizer();
        const transferRecognizer = makeTransferRecognizer(baseRecognizer)

        for (const command of commands) { // await does not work in forEach
            await getSound(transferRecognizer, command, numberOfSamples);
        }

        await getSound(transferRecognizer, '_background_noise_', numberOfSamples);
        console.table(transferRecognizer.countExamples());
        console.table(transferRecognizer.getMetadata())
        setActionText(`Training...`);
        await transferRecognizer.train({
            epochs: 25,
            callback: {
                onEpochEnd: async (epoch, logs) => {
                    console.log(`Epoch ${epoch}: loss=${logs?.loss}, accuracy=${logs?.acc}`);
                },
            },
        });
        console.log("modelInputShape makeModel", transferRecognizer.modelInputShape())
        setActionText(`Training completed`);
        setTransferRecognizer(transferRecognizer);
    }

    const runModel = async () => {
        setActionText('Listening... ');
        if (transferRecognizer !== null) {
            console.log("modelInputShape runModel", transferRecognizer.modelInputShape())
            console.log("transferRecognizer.modelInputShape()[1]", transferRecognizer.modelInputShape()[1]  )

            console.log("wordLabels  runModel", transferRecognizer.wordLabels().sort())
            await transferRecognizer.listen(
                // @ts-ignore
                result => {
                    const words = transferRecognizer.wordLabels().sort();
                    // @ts-ignore
                    const scores = Array.from(result.scores).map((s, i) => ({
                        score: s,
                        word: words[i],
                    }));

                    scores.sort((s1, s2) => s2.score - s1.score);
                    setActionText(`Score for word '${scores[0].word}' = ${scores[0].score}`);
                },
                {
                    includeSpectrogram: true,
                    probabilityThreshold: 0.75,
                    invokeCallbackOnNoiseAndUnknown: true,
                    overlapFactor: 0.50
                }
            );
            setTimeout(() => {
                transferRecognizer.stopListening();
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
            if (transferRecognizer !== null) {
                const serialized = transferRecognizer.serializeExamples();
                saveByteArray([serialized], "trainedmodel.txt");
            }
        }

        const makeTransferRecognizer = (br: SpeechCommandRecognizer) =>
            br.createTransfer("navigation")

        async function loadDatasetInTransferRecognizer(serialized: string | ArrayBuffer | null | undefined) {
            const br = await makeBaseRecognizer();
            const tr = makeTransferRecognizer(br)

            if (serialized !== null && typeof serialized !== "undefined") {
                tr.loadExamples(serialized as ArrayBuffer, false)
                const exampleCounts = tr.countExamples();
                const transferWords = [];
                console.log("modelInputShape load dataset", tr.modelInputShape())
                const modelNumFrames = tr.modelInputShape()[1];
                const durationMultipliers = [];
                for (const word in exampleCounts) {
                    transferWords.push(word);
                    const examples = tr.getExamples(word);
                    for (const example of examples) {
                        const spectrogram = example.example.spectrogram;
                        // Ignore _background_noise_ examples when determining the duration
                        // multiplier of the dataset.
                        if (word !== BACKGROUND_NOISE_TAG && modelNumFrames !== null) {
                            durationMultipliers.push(Math.round(
                                spectrogram.data.length / spectrogram.frameSize / modelNumFrames));
                        }
                    }
                }
                setCommands(transferWords);
                setCommandsAsText(transferWords.join(", "));

                // Determine the transferDurationMultiplier value from the dataset.
                const transferDurationMultiplier =
                    durationMultipliers.length > 0 ? Math.max(...durationMultipliers) : 1;
                console.log(
                    `Deteremined transferDurationMultiplier from uploaded ` +
                    `dataset: ${transferDurationMultiplier}`);
                setTransferRecognizer(tr);
            }
        }

        const uploadModel = () => {
            const files = fileInput.current?.files;
            if (files == null || files.length !== 1) {
                throw new Error('Must select exactly one file.');
            }
            const datasetFileReader = new FileReader();
            datasetFileReader.onload = async event => {
                try {
                    await loadDatasetInTransferRecognizer(event.target?.result);
                } catch (err) {

                }
            };
            datasetFileReader.onerror = () =>
                console.error(`Failed to get binary data from file '${files[0].name}'.`);
            datasetFileReader.readAsArrayBuffer(files[0]);
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
                        <button disabled={transferRecognizer === null} onClick={runModel}>Run model</button>
                        <button disabled={transferRecognizer === null} onClick={downloadModel}>Download model</button>
                        <div>
                            <label>
                                Upload file:
                                <input type="file" ref={fileInput}/>
                            </label>
                            <button onClick={uploadModel}>Upload model</button>
                        </div>
                    </div>
                </main>
            </div>
        );
    }

    export default App;
