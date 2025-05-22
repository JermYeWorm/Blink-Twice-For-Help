# Blink Twice For Help: A Brain-Inspired Spike Encoder and SNN SoC for Continuous Cognitive State Monitoring

## Abstract
Sleep quality, attentional state, and real-time cogni
tive assessment are increasingly central to both clinical research
and consumer health technologies. Conventional methods for
monitoring these parameters often lack the temporal resolution,
accuracy, or adaptability required for real-time applications. Eye
movements, specifically blink duration and intensity, provide
a promising, non-invasive biosignal for continuous tracking of
cognitive states. In this work, we present a mixed-signal neuromorphic System-
on-Chip (SoC) designed for real-time, low-power decoding of
blink-based electroencephalogram (EEG) biosignals. Designed
using the open-source 1.8V SkyWater 130nm CMOS process,
the system integrates analog front-end circuitry with delta-
modulation-based spike encoding and on-chip classification via a
Spiking Neural Network (SNN). Our design supports applications
in sleep and attention monitoring, cognitive workload analysis,
and potential neuromodulation therapies for neurodegenerative
and mental health disorders. The platform operates with high
computational efficiency, making it ideal for future scaling into
wearable and embedded systems. System-level validation was
conducted using full-custom layout, simulation with real EEG
blink data, and verification via the Cadence Design Suite. The
final chip occupies 0.742 mm^2 of a 1.6 mm × 1.6 mm die,
consisting of 276,542 transistors, and consumes 63.5 μW in the
analog domain and 43 μW in the digital domain.

![alt text](https://github.com/JermYeWorm/Neural_Interface_Chip/blob/main/full_chip_layout.png?raw=true)
